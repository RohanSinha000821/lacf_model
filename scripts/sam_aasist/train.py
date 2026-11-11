"""Fresh SAM-AASIST on one fold's source train/dev only; no resume or target access."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, compute_eer, compute_auroc, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.sota.aasist import AASISTDataset, make_eval_loader, make_train_loader
from audio_deepfake_detection.sota.sam_aasist import (
    SAMAASIST, MODEL_CONFIG, PARAMETER_COUNT, SAM_UPSTREAM_COMMIT, POOL_RATIO_STATUS,
)
from audio_deepfake_detection.sota.sam_optimization import SharpnessAwareStep
from scripts.aasist.train import (
    BATCH_SIZE, CLASS_WEIGHTS, MAX_EPOCHS, PATIENCE, LEARNING_RATE, MIN_LEARNING_RATE,
    WEIGHT_DECAY, make_optimizer, set_seed, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILY = "sam_aasist"
SEED_PLAN = (1234, 2345, 3456)
RHO = 0.05
RUNTIME_FILES = (
    "scripts/sam_aasist/train.py", "scripts/aasist/train.py",
    "src/audio_deepfake_detection/sota/sam_aasist.py",
    "src/audio_deepfake_detection/sota/sam_optimization.py",
    "src/audio_deepfake_detection/sota/aasist.py",
    "src/audio_deepfake_detection/sota/aasist_arch.py",
    "src/audio_deepfake_detection/data.py", "src/audio_deepfake_detection/protocol.py",
    "src/audio_deepfake_detection/metrics.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, choices=SEED_PLAN, default=1234)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.num_workers, args.eval_workers) < 0 or min(args.prefetch_factor, args.cpu_threads) < 1:
        parser.error("Workers must be nonnegative; prefetch and CPU threads positive")
    return args


def source_guards():
    return {p: checkpoint_digest(ROOT / p) for p in RUNTIME_FILES}


def check_guards(config):
    if source_guards() != config["source_sha256"]:
        raise RuntimeError("Guarded runtime changed; stop without retry")


def source_records(fold, data_root):
    return {d: {s: read_dataset(d, s, data_root) for s in ("train", "dev")} for d in FOLDS[fold]["sources"]}


def source_identity(records, data_root):
    result = {}
    for domain, splits in records.items():
        result[domain] = {}
        for split, rows in splits.items():
            identities = [(utterance_id(p, domain, data_root), int(y)) for p, y in rows]
            if not identities or len({p for p, _ in identities}) != len(rows) or any(y not in (0, 1) for _, y in identities):
                raise ValueError(f"Invalid canonical source membership: {domain}/{split}")
            payload = json.dumps(identities, separators=(",", ":")).encode()
            result[domain][split] = {"count": len(rows), "ordered_ids_labels_sha256": hashlib.sha256(payload).hexdigest(),
                                     "bona_fide": sum(y == 0 for _, y in identities), "spoof": sum(y == 1 for _, y in identities)}
    return result


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Preserving interrupted checkpoint: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def score_sources(model, loaders, device):
    model.eval()
    result = {}
    for domain, loader in loaders.items():
        labels = torch.empty(len(loader.dataset), dtype=torch.long)
        scores = torch.empty(len(loader.dataset), device=device, dtype=torch.float32)
        count = 0
        started = time.monotonic()
        for waves, y in loader:
            if len(y) != 1:
                raise ValueError("Source scoring requires batch one")
            scores[count] = model.spoof_score(model(waves.to(device, non_blocking=True)))[0].float()
            labels[count] = y[0]
            count += 1
            if count % 5000 == 0:
                print(f"dev {domain}: {count}/{len(labels)}; {count/(time.monotonic()-started):.1f} audio/s", flush=True)
        scores = scores.cpu()
        if count != len(labels) or not torch.isfinite(scores).all():
            raise RuntimeError("Incomplete or nonfinite source scores")
        result[domain] = (labels.numpy(), scores.numpy())
    return result


def source_metrics(scored):
    by_domain = {d: {"eer": compute_eer(y, s), "auroc": compute_auroc(y, s)} for d, (y, s) in scored.items()}
    return {"by_domain": by_domain, "macro_eer": float(np.mean([m["eer"] for m in by_domain.values()]))}


def train_epoch(sam, loader, device, *, snapshot=lambda step, seen: None, log=print):
    sam.model.train()
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    seen, data_wait = 0, 0.0
    started = last_log = last_snapshot = time.monotonic()
    iterator = iter(loader)
    for step in range(1, len(loader) + 1):
        tick = time.monotonic()
        waves, labels = next(iterator)
        data_wait += time.monotonic() - tick
        loss, perturbed = sam.step(waves.to(device, non_blocking=True), labels.to(device, non_blocking=True))
        seen += len(labels)
        sums += torch.stack((loss, perturbed)).double() * len(labels)
        now = time.monotonic()
        if now - last_log >= 30 or step == len(loader):
            values = (sums / seen).tolist()
            peak = torch.cuda.max_memory_allocated(device)/1024**3 if torch.device(device).type == "cuda" else 0
            log(f"step={step}/{len(loader)} loss={values[0]:.6f} perturbed={values[1]:.6f} "
                f"audio/s={seen/(now-started):.2f} host_data_wait={data_wait:.1f}s "
                f"ETAmin={(len(loader)-step)*(now-started)/step/60:.1f} peak_alloc_GiB={peak:.3f}", flush=True)
            last_log = now
        if now-last_snapshot >= 900 or step == len(loader):
            snapshot(step, seen)
            last_snapshot = time.monotonic()
    if not seen:
        raise ValueError("Empty training loader")
    values = (sums/seen).tolist()
    return {"loss": values[0], "perturbed_loss": values[1], "examples": seen,
            "seconds": time.monotonic()-started, "host_data_wait_seconds": data_wait}


def fit(model, train, dev, device, run, config, *, score=score_sources, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, log=print, guard=check_guards):
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/"config.json", config)
    optimizer, scheduler = make_optimizer(model, steps_per_epoch=len(train))
    sam = SharpnessAwareStep(model, optimizer, scheduler,
                             torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device)), rho=RHO)
    best, best_epoch, stale = float("inf"), 0, 0
    started = time.monotonic()
    for epoch in range(1, max_epochs+1):
        guard(config)
        log(f"Epoch {epoch} training starts", flush=True)
        def snapshot(step, seen):
            if sam.perturbed:
                raise RuntimeError("Refusing perturbed checkpoint")
            save_checkpoint(run/"recovery.pt", {"config": config, "epoch": epoch, "step": step,
                "recordings_seen": seen, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "sam_state": sam.state_dict(), "kind": "intermediate_snapshot_not_completion",
                "exact_mid_epoch_resume_supported": False})
            log(f"SNAPSHOT epoch={epoch} step={step}; not a completion marker", flush=True)
        losses = train_epoch(sam, train, device, snapshot=snapshot, log=log)
        tick = time.monotonic()
        scored = score(model, dev, device)
        metrics = source_metrics(scored)
        if not np.isfinite(metrics["macro_eer"]):
            raise RuntimeError("Nonfinite source macro EER")
        improved = metrics["macro_eer"] < best
        if improved:
            best, best_epoch, stale = metrics["macro_eer"], epoch, 0
        else:
            stale += 1
        if sam.perturbed:
            raise RuntimeError("Refusing perturbed checkpoint")
        payload = {"config": config, "fold": config["fold"], "epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "sam_state": sam.state_dict(),
            "source_dev": metrics, "macro_dev_eer": metrics["macro_eer"], "best_macro_dev_eer": best,
            "bad_epochs": stale, "optimizer_updates": epoch*len(train), "train": losses,
            "dev_seconds": time.monotonic()-tick,
            "source_dev_scores": {d: torch.from_numpy(s.copy()) for d, (_, s) in scored.items()}}
        for name in (("last.pt", "best.pt") if improved else ("last.pt",)):
            save_checkpoint(run/name, payload)
        with (run/"history.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train": losses, "source_dev": metrics,
                                "dev_seconds": payload["dev_seconds"], "best_epoch": best_epoch, "patience": stale})+"\n")
        log(f"EPOCH {epoch} macro_source_EER={metrics['macro_eer']:.8f} "
            f"best_epoch={best_epoch} patience={stale}/{patience} train_s={losses['seconds']:.1f} "
            f"dev_s={payload['dev_seconds']:.1f}", flush=True)
        if stale >= patience:
            break
    guard(config)
    result = {"fold": config["fold"], "seed": config["seed"], "best_epoch": best_epoch, "final_epoch": epoch,
              "best_macro_source_dev_eer": best, "wall_seconds": time.monotonic()-started, "target_access": False}
    if torch.device(device).type == "cuda":
        result.update(peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result  # Completion requires a separate fresh selected-checkpoint/source reload below.


def finalize(run, config, records, dev, device, completion, *, model_factory=SAMAASIST, score=score_sources):
    check_guards(config)
    checkpoint = torch.load(run/"best.pt", map_location="cpu", weights_only=True)
    fresh = model_factory().to(device).eval()
    fresh.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scored = score(fresh, dev, device)
    if source_metrics(scored) != checkpoint["source_dev"] or any(
            not np.array_equal(s, checkpoint["source_dev_scores"][d].numpy()) for d, (_, s) in scored.items()):
        raise RuntimeError("Selected-checkpoint source reload differs; no completion marker")
    hashes = {}
    for domain, (labels, scores) in scored.items():
        if list(labels) != [y for _, y in records[domain]["dev"]]:
            raise RuntimeError("Source label/order mismatch")
        path = run/f"source_dev_{domain}.csv"
        write_score_csv(path, ((utterance_id(p, domain, config["data_root"]), domain, y, float(s))
                              for (p, y), s in zip(records[domain]["dev"], scores)))
        hashes[path.name] = checkpoint_digest(path)
    write_json(run/"source_calibration.json", calibrate_source_scores(scored))
    check_guards(config)
    completion.update(best_checkpoint_sha256=checkpoint_digest(run/"best.pt"), source_scores_sha256=hashes,
                      source_calibration_sha256=checkpoint_digest(run/"source_calibration.json"),
                      selected_checkpoint_reloaded_and_source_verified=True)
    write_json(run/"training_complete.json", completion)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no automatic precision/batch fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    run = ROOT/"outputs"/FAMILY/args.fold/str(args.seed)
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent/f".{args.seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run.exists():
            raise FileExistsError(f"Refusing existing run: {run}")
        set_seed(args.seed)
        records = source_records(args.fold, args.data_root)
        identity = source_identity(records, args.data_root)
        train = make_train_loader([AASISTDataset(parts["train"], training=True) for parts in records.values()],
                                  batch_size=BATCH_SIZE, seed=args.seed, num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch_factor)
        dev = {d: make_eval_loader(AASISTDataset(parts["dev"], training=False), num_workers=args.eval_workers,
                                   prefetch_factor=args.prefetch_factor) for d, parts in records.items()}
        model = SAMAASIST().cuda()
        if sum(p.numel() for p in model.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("Unexpected parameter count")
        config = {**vars(args), "data_root": str(args.data_root), "run_name": FAMILY, "model_name": "SAM-AASIST",
            "model_config": MODEL_CONFIG, "sam_upstream_commit": SAM_UPSTREAM_COMMIT,
            "pool_ratio_status": POOL_RATIO_STATUS, "source_identity": identity,
            "sources": list(records), "target": FOLDS[args.fold]["target"], "target_access": False,
            "dataset_counts": {d: {s: len(v) for s, v in parts.items()} for d, parts in records.items()},
            "seed_plan": list(SEED_PLAN), "initialization": "random fresh; no pretrained/pilot/fold weights",
            "parameter_count": PARAMETER_COUNT, "trainable_parameters": PARAMETER_COUNT,
            "batch_size": BATCH_SIZE, "gradient_accumulation_steps": 1, "eval_batch_size": 1,
            "precision": "float32", "cuda_matmul_tf32": False, "cudnn_tf32": True,
            "optimizer": "nonadaptive_SAM_Adam", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "rho": RHO, "class_weights": list(CLASS_WEIGHTS),
            "scheduler": "per_Adam_update_cosine", "minimum_learning_rate": MIN_LEARNING_RATE,
            "scheduler_total_steps": MAX_EPOCHS*len(train), "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "selection": "strict_equal_domain_macro_source_dev_eer_ties_earlier",
            "sampler": "equal_domain_natural_classes_replacement_N_draws",
            "train_input": "random64600_after16k_resampling_repeat_short", "eval_input": "first64600_repeat_short",
            "rawboost_algorithm": 0, "freq_aug": False, "swa": False, "label_order": ["bona_fide", "spoof"],
            "score": "spoof_logit_minus_bona_fide_logit", "source_sha256": source_guards(),
            "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                                  env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip(),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "cudnn_deterministic": True, "cudnn_benchmark": False, "pin_memory": True,
            "persistent_workers": args.num_workers > 0, "training_resume": False,
            "snapshot_policy": "15min_and_end_train_phase; not exact resume or completion"}
        print(f"START {FAMILY}/{args.fold}/{args.seed}; {PARAMETER_COUNT} trainables; batch24 FP32; "
              f"2 forward/backward passes per update; train={len(train.dataset)}; targets unopened", flush=True)
        torch.cuda.reset_peak_memory_stats()
        completion = fit(model, train, dev, "cuda", run, config)
        model.cpu()
        del model
        torch.cuda.empty_cache()
        finalize(run, config, records, dev, "cuda", completion)
        print(f"COMPLETE {run}; full source reload/calibration verified; target unopened", flush=True)


if __name__ == "__main__":
    main()
