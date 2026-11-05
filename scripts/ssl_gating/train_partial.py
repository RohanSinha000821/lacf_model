"""Fresh source-only SSL top-four/BF16 variant; no target access or automatic resume."""

import argparse
import fcntl
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, compute_auroc, compute_eer, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, make_eval_loader, make_train_loader
from audio_deepfake_detection.sota.ssl_gating_partial import (
    PARTIAL_MODEL_CONFIG, RUN_FAMILY, PartialSSLFeatureGating, detection_losses, partial_scores,
)
from scripts.ssl_gating.train import LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, set_seed, write_json


ROOT = Path(__file__).resolve().parents[2]


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Preserving interrupted checkpoint: {temporary}")
    torch.save(value, temporary)
    temporary.replace(path)


def source_guards():
    paths = [Path(__file__), ROOT / "src/audio_deepfake_detection/sota/ssl_gating_partial.py",
             ROOT / "scripts/ssl_gating/train.py",
             *[ROOT / f"src/audio_deepfake_detection/sota/{n}.py" for n in ("ssl_gating", "ssl_gating_blocks")],
             *[ROOT / f"src/audio_deepfake_detection/{n}.py" for n in ("data", "protocol", "metrics")]]
    return {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in paths}


@torch.inference_mode()
def score_sources(model, loaders):
    model.eval()
    result = {}
    for domain, loader in loaders.items():
        scores = torch.empty(len(loader.dataset), device="cuda", dtype=torch.float32)
        labels = torch.empty(len(loader.dataset), dtype=torch.long)
        tick = time.monotonic()
        for i, (waves, y) in enumerate(loader):
            scores[i] = partial_scores(model, waves.cuda(non_blocking=True))[0]
            labels[i] = y[0]
            if (i + 1) % 5000 == 0:
                print(f"dev {domain}: {i+1}/{len(scores)}; {(i+1)/(time.monotonic()-tick):.2f} audio/s", flush=True)
        scores = scores.cpu()
        if i+1 != len(loader.dataset) or not torch.isfinite(scores).all():
            raise RuntimeError("Incomplete/nonfinite full-recording source scores")
        result[domain] = (labels.numpy(), scores.numpy())
    return result


def source_metrics(scored):
    by_domain = {d: {"eer": compute_eer(y, s), "auroc": compute_auroc(y, s)} for d, (y, s) in scored.items()}
    return {"by_domain": by_domain,
            "macro_eer": sum(v["eer"] for v in by_domain.values()) / len(by_domain),
            "macro_auroc": sum(v["auroc"] for v in by_domain.values()) / len(by_domain)}


def train_epoch(model, loader, optimizer, epoch, snapshot, checkpoint_seconds):
    model.train()
    total = torch.zeros(3, device="cuda", dtype=torch.float64)
    seen = 0
    started = last_log = last_snapshot = time.monotonic()
    for step, (waves, labels) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        loss, ce, cka = detection_losses(model, waves.cuda(non_blocking=True), labels.cuda(non_blocking=True))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite BF16/FP32 loss; no fallback or retry")
        loss.backward()
        optimizer.step()
        seen += len(labels)
        total += torch.stack((loss.detach(), ce.detach(), cka.detach())).double() * len(labels)
        now = time.monotonic()
        if now-last_log >= 30 or step == len(loader):
            values = (total/seen).tolist()
            print(f"epoch={epoch} step={step}/{len(loader)} recordings={seen}/{len(loader.dataset)} "
                  f"loss={values[0]:.6f} ce={values[1]:.6f} cka={values[2]:.6f} "
                  f"audio/s={seen/(now-started):.2f} ETAmin={(len(loader.dataset)-seen)/max(seen,1)*(now-started)/60:.1f}", flush=True)
            last_log = now
        if now-last_snapshot >= checkpoint_seconds or step == len(loader):
            snapshot(epoch, step, seen)
            last_snapshot = time.monotonic()
    return dict(zip(("total", "ce", "cka"), (total/seen).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=19)
    parser.add_argument("--checkpoint-minutes", type=float, default=15)
    args = parser.parse_args()
    if (args.seed < 0 or args.batch_size < 2 or min(args.num_workers,args.eval_workers) < 0
            or args.prefetch_factor < 1 or not math.isfinite(args.checkpoint_minutes) or args.checkpoint_minutes <= 0
            or not math.isfinite(args.gpu_memory_budget_gib) or args.gpu_memory_budget_gib <= 0):
        parser.error("Invalid batch/seed/worker/checkpoint or memory settings")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no precision fallback")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run = ROOT / "outputs" / RUN_FAMILY / args.fold / str(args.seed)
    run.mkdir(parents=True, exist_ok=False)
    with (run / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        set_seed(args.seed)
        sources = FOLDS[args.fold]["sources"]
        records = {d: {s: read_dataset(d,s,args.data_root) for s in ("train","dev")} for d in sources}
        train = make_train_loader([FullAudioDataset(records[d]["train"],training=True) for d in sources],
                                  batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch_factor)
        dev = {d: make_eval_loader(FullAudioDataset(records[d]["dev"]), num_workers=args.eval_workers,
                                   prefetch_factor=args.prefetch_factor) for d in sources}
        model = PartialSSLFeatureGating().cuda()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        config = {**vars(args), "data_root": str(args.data_root), "run_name": RUN_FAMILY,
                  "model_name": "PartialSSLFeatureGating", "model_config": PARTIAL_MODEL_CONFIG,
                  "frontend_config": model.feature_extractor.config.to_dict(), "sources": list(sources),
                  "dataset_counts": {d: {s:len(v) for s,v in parts.items()} for d,parts in records.items()},
                  "target_domain": FOLDS[args.fold]["target"], "target_access": False,
                  "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "scheduler": "constant",
                  "optimizer": "Adam", "loss": "weighted_CE_0.9_0.1_plus_CKA_coefficient1",
                  "precision": "BF16_autocast_FP32_pooling_losses_scores", "effective_batch": args.batch_size,
                  "accumulation_steps": 1, "train_max_seconds": 20, "eval_batch_size": 1,
                  "selection": "strict_equal_domain_macro_source_dev_eer_ties_earlier",
                  "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "total_parameters": sum(p.numel() for p in model.parameters()),
                  "source_sha256": source_guards(), "git_revision": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                  "initialization": "fresh official pinned XLS-R plus fresh backend; no interrupted weights",
                  "snapshot_policy": f"atomic every{args.checkpoint_minutes}min_and_end_of_train_epoch; no automatic or exact mid-epoch resume",
                  "transformers": __import__("transformers").__version__,
                  "numpy": __import__("numpy").__version__, "python": __import__("platform").python_version()}
        write_json(run/"config.json",config)
        print(f"START {RUN_FAMILY}/{args.fold}/{args.seed}; sources={sources}; target unopened; "
              f"trainables={config['trainable_parameters']} batch={args.batch_size} BF16/FP32",flush=True)
        def snapshot(epoch, step, seen):
            save_checkpoint(run/"recovery.pt", {"config": config,"epoch": epoch,"step":step,"recordings_seen":seen,
                "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                "kind":"intermediate_training_snapshot_not_completion","exact_mid_epoch_resume_supported":False})
            print(f"SNAPSHOT recovery.pt epoch={epoch} step={step}; not a completion marker",flush=True)
        best, best_epoch, stale, best_scores = math.inf, 0, 0, None
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, MAX_EPOCHS+1):
            if source_guards() != config["source_sha256"]:
                raise RuntimeError("Training runtime changed; stop without retry")
            losses = train_epoch(model,train,optimizer,epoch,snapshot,args.checkpoint_minutes*60)
            scored = score_sources(model,dev)
            metrics = source_metrics(scored)
            if metrics["macro_eer"] < best:
                best,best_epoch,stale = metrics["macro_eer"],epoch,0
                best_scores = {d: s.copy() for d,(_,s) in scored.items()}
                names = ("last.pt","best.pt")
            else:
                stale += 1
                names = ("last.pt",)
            payload = {"config":config,"fold":args.fold,"epoch":epoch,"model_state_dict":model.state_dict(),
                       "optimizer_state_dict":optimizer.state_dict(),"source_dev":metrics,"macro_dev_eer":metrics["macro_eer"],
                       "train_losses":losses,"best_macro_dev_eer":best,"bad_epochs":stale}
            for name in names:
                save_checkpoint(run/name,payload)
            with (run/"history.jsonl").open("a") as f:
                f.write(json.dumps({"epoch":epoch,"losses":losses,"source_dev":metrics,"best_epoch":best_epoch,"patience":stale})+"\n")
            print(f"EPOCH {epoch} macro_source_EER={100*metrics['macro_eer']:.4f}% best={best_epoch} patience={stale}/{PATIENCE}",flush=True)
            if stale >= PATIENCE:
                break
        checkpoint = torch.load(run/"best.pt",map_location="cpu",weights_only=True)
        model.cpu()
        del optimizer,model,payload
        torch.cuda.empty_cache()
        fresh = PartialSSLFeatureGating.from_frontend_config(config["frontend_config"]).cuda().eval()
        fresh.load_state_dict(checkpoint["model_state_dict"],strict=True)
        reloaded = score_sources(fresh,dev)
        if source_metrics(reloaded) != checkpoint["source_dev"] or any(not (reloaded[d][1] == best_scores[d]).all() for d in sources):
            raise RuntimeError("Selected-checkpoint full-source reload differs; no completion marker")
        hashes = {}
        for d,(labels,scores) in reloaded.items():
            path = run/f"source_dev_{d}.csv"
            write_score_csv(path,((utterance_id(p,d,args.data_root),d,y,float(s)) for (p,y),s in zip(records[d]["dev"],scores)))
            hashes[path.name] = checkpoint_digest(path)
        write_json(run/"source_calibration.json",calibrate_source_scores(reloaded))
        write_json(run/"training_complete.json",{"fold":args.fold,"seed":args.seed,"best_epoch":best_epoch,
            "final_epoch":epoch,"best_macro_source_dev_eer":best,"best_checkpoint_sha256":checkpoint_digest(run/"best.pt"),
            "source_scores_sha256":hashes,"source_calibration_sha256":checkpoint_digest(run/"source_calibration.json"),
            "selected_checkpoint_reloaded_and_source_verified":True,"wall_seconds":time.monotonic()-started,"target_access":False})
        print("COMPLETE: full source reload/calibration passed; no target access",flush=True)


if __name__ == "__main__":
    main()
