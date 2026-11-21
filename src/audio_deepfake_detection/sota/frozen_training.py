"""Source-only fitting for the explicitly adapted LHCC/NSF-PINN frozen detectors.

No target reader, resume, multi-fold queue or implicit weight download. Native
fixed-length audio inputs differ by method. Official encoders remain eval/no-grad.
"""

import argparse
from dataclasses import asdict
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

from audio_deepfake_detection.data import load_audio_segment, resample_audio
from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FINAL_SEEDS, FOLDS, checkpoint_digest, utterance_id
from audio_deepfake_detection.sota.aasist import make_eval_loader, make_train_loader
from audio_deepfake_detection.sota.frozen_frontend import FrozenSpeechEncoder, pretrained_identity
from audio_deepfake_detection.sota.lhcc import LHCCConfig, LHCCDetector, classification_loss
from audio_deepfake_detection.sota.nsf_pinn import NSFPINNConfig, NSFPINNDetector, physics_loss
from audio_deepfake_detection.sota.sam_optimization import SharpnessAwareStep
from scripts.aasist.train import set_seed, write_json
from scripts.sam_aasist.train import source_identity, source_metrics, source_records, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
RECIPES = {
    "lhcc": {"family": "lhcc_frozen_v1", "samples": 64600, "batch_size": 8,
             "learning_rate": 1e-6, "weight_decay": 1e-4, "max_epochs": 10, "patience": 2,
             "optimizer": "Adam", "scheduler": "constant", "rho": None,
             "train_input": "first64600_repeat_short", "dev_input": "first64600_repeat_short"},
    "nsf_pinn": {"family": "nsf_pinn_frozen_v1", "samples": 64000, "batch_size": 32,
                 "learning_rate": 3e-4, "weight_decay": 1e-4, "max_epochs": 100, "patience": 7,
                 "optimizer": "SAM_AdamW", "scheduler": "per_update_cosine_to_zero", "rho": 0.05,
                 "train_input": "random64000_zero_pad_short", "dev_input": "center64000_zero_pad_short"},
}
DETECTORS = {"lhcc": LHCCDetector, "nsf_pinn": NSFPINNDetector}


def fixed_audio(wave, method, *, training):
    if wave.ndim != 1 or not wave.numel() or not torch.isfinite(wave).all():
        raise ValueError("Expected finite, nonempty mono audio")
    size = RECIPES[method]["samples"]
    if len(wave) < size:
        if method == "lhcc":
            return wave.repeat((size + len(wave) - 1) // len(wave))[:size]
        return F.pad(wave, (0, size - len(wave)))
    if len(wave) == size:
        return wave.contiguous()
    start = 0
    if method == "nsf_pinn":
        start = int(torch.randint(len(wave) - size + 1, ()).item()) if training else (len(wave) - size) // 2
    return wave[start:start + size].contiguous()


def load_fixed_audio(path, method, *, training):
    """Exact 16kHz seek/read fast path; other rates retain full-resample-then-crop.

    Cropping before resampling changes boundary filters. Never apply the seek
    optimization at a different rate, and never shorten the model's native view.
    """
    with sf.SoundFile(path) as audio:
        if audio.samplerate == 16000:
            size, length = RECIPES[method]["samples"], len(audio)
            start = 0
            if length > size and method == "nsf_pinn":
                start = int(torch.randint(length - size + 1, ()).item()) if training else (length - size) // 2
            audio.seek(start)
            wave = torch.from_numpy(audio.read(frames=min(size,length), dtype="float32", always_2d=True)).mean(dim=1)
            return fixed_audio(wave, method, training=training)
    wave, rate = load_audio_segment(path)
    return fixed_audio(resample_audio(wave, rate, 16000).float(), method, training=training)


class DetectorAudioDataset(Dataset):
    def __init__(self, records, method, *, training):
        if not records or method not in RECIPES:
            raise ValueError("Nonempty source records and a supported method required")
        self.records, self.method, self.training = records, method, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        if label not in (0, 1):
            raise ValueError("Invalid canonical source label")
        return load_fixed_audio(path, self.method, training=self.training), label


class TrainingHead(nn.Module):
    """SAM sees only detector parameters and reuses one detached encoder result."""
    def __init__(self, detector, method):
        super().__init__()
        self.detector, self.method = detector, method

    def forward(self, features):
        if self.method == "nsf_pinn":
            return self.detector(features, return_physics=True)
        return self.detector(features)


def make_update(detector, method, *, total_steps, epoch=1):
    recipe = RECIPES[method]
    head = TrainingHead(detector, method)
    optimizer_cls = torch.optim.AdamW if method == "nsf_pinn" else torch.optim.Adam
    optimizer = optimizer_cls(head.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    if method == "nsf_pinn":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
        criterion = lambda output, y: physics_loss(output, y, detector.cfg, epoch=epoch)[0]
        sam = SharpnessAwareStep(head, optimizer, scheduler, criterion, rho=recipe["rho"])
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        sam = None
    return head, optimizer, scheduler, sam


def update_batch(head, optimizer, scheduler, sam, features, labels, *, epoch):
    head.train()
    if sam is not None:
        sam.criterion = lambda output, y: physics_loss(output, y, head.detector.cfg, epoch=epoch)[0]
        first, second = sam.step(features, labels)
        return torch.stack((first, second))
    optimizer.zero_grad(set_to_none=True)
    loss = classification_loss(head(features), labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float("inf"), error_if_nonfinite=True)
    if not bool(torch.isfinite(torch.stack((loss.detach(), norm))).all()):
        raise RuntimeError("Nonfinite detector update; stopped without retry")
    optimizer.step()
    scheduler.step()
    return torch.stack((loss.detach(), loss.detach()))


@torch.no_grad()
def score_sources(encoder, detector, loaders, device):
    encoder.eval()
    detector.eval()
    result = {}
    for domain, loader in loaders.items():
        labels = torch.empty(len(loader.dataset), dtype=torch.long)
        scores = torch.empty(len(loader.dataset), device=device, dtype=torch.float32)
        count, started = 0, time.monotonic()
        for waves, y in loader:
            if len(y) != 1:
                raise ValueError("Batch-one development required until parity validation")
            features = encoder(waves.to(device, non_blocking=True)).float()
            scores[count] = detector.spoof_score(detector(features))[0].float()
            labels[count] = y[0]
            count += 1
            if count % 5000 == 0:
                print(f"dev {domain} {count}/{len(labels)} {count/(time.monotonic()-started):.2f} audio/s", flush=True)
        if count != len(labels) or not torch.isfinite(scores).all():
            raise RuntimeError("Incomplete or nonfinite source scores")
        result[domain] = (labels.numpy(), scores.cpu().numpy())
    return result


def source_guards(method):
    paths = [f"scripts/{method}/train.py", f"scripts/{method}/README.md",
             "src/audio_deepfake_detection/sota/frozen_frontend.py",
             "src/audio_deepfake_detection/sota/frozen_training.py", f"src/audio_deepfake_detection/sota/{method}.py",
             "src/audio_deepfake_detection/sota/sam_optimization.py", "scripts/sam_aasist/train.py",
             "scripts/aasist/train.py", "src/audio_deepfake_detection/sota/aasist.py",
             "src/audio_deepfake_detection/sota/aasist_arch.py", "src/audio_deepfake_detection/data.py",
             "src/audio_deepfake_detection/protocol.py", "src/audio_deepfake_detection/metrics.py"]
    return {p: checkpoint_digest(ROOT / p) for p in paths}


def check_guards(config):
    if source_guards(config["method"]) != config["source_sha256"]:
        raise RuntimeError("Runtime sources changed; stop without retry")


def preflight(encoder, detector, method, train, dev, device):
    """Discard one real-source update, then prove independent detector reload parity."""
    head, optimizer, scheduler, sam = make_update(detector, method, total_steps=len(train) * RECIPES[method]["max_epochs"])
    waves, labels = next(iter(train))
    features = encoder(waves.to(device)).float()
    losses = update_batch(head, optimizer, scheduler, sam, features, labels.to(device), epoch=1)
    small = {d: make_eval_loader(Subset(loader.dataset, sorted({0, len(loader.dataset) - 1})), num_workers=0)
             for d, loader in dev.items()}
    expected = score_sources(encoder, detector, small, device)
    fresh = DETECTORS[method](detector.cfg).to(device)
    fresh.load_state_dict(detector.state_dict(), strict=True)
    actual = score_sources(encoder, fresh, small, device)
    if any(not np.array_equal(expected[d][1], actual[d][1]) for d in expected):
        raise RuntimeError("Discarded source preflight reload mismatch")
    if any(p.grad is not None or p.requires_grad for p in encoder.parameters()) or encoder.training:
        raise RuntimeError("Frozen encoder invariant failed")
    return {"finite_update_losses": losses.tolist(), "first_tail_source_reload_bit_exact": True,
            "full_training_batch_size": len(labels), "probe_weights_discarded": True,
            "target_access": False, "scope": "bounded source probe; not full-source final validation"}


def fit(encoder, detector, method, train, dev, device, run, config):
    recipe = RECIPES[method]
    head, optimizer, scheduler, sam = make_update(detector, method, total_steps=len(train) * recipe["max_epochs"])
    best, best_epoch, stale, updates = float("inf"), 0, 0, 0
    started = time.monotonic()
    for epoch in range(1, recipe["max_epochs"] + 1):
        check_guards(config)
        sums, seen = torch.zeros(2, device=device, dtype=torch.float64), 0
        tick = last_log = last_snapshot = time.monotonic()
        for step, (waves, labels) in enumerate(train, 1):
            # Exactly one frozen forward per batch, including SAM's two detector passes.
            features = encoder(waves.to(device, non_blocking=True)).float()
            losses = update_batch(head, optimizer, scheduler, sam, features, labels.to(device), epoch=epoch)
            sums += losses.double() * len(labels)
            seen += len(labels)
            updates += 1
            now = time.monotonic()
            if now - last_log >= 30 or step == len(train):
                print(f"epoch={epoch} step={step}/{len(train)} loss={(sums/seen).tolist()} "
                      f"audio/s={seen/(now-tick):.2f} ETAmin={(len(train)-step)*(now-tick)/step/60:.1f}", flush=True)
                last_log = now
            if now - last_snapshot >= 900:
                save_checkpoint(run/"snapshot.pt", {"config": config, "epoch": epoch, "step": step,
                    "detector_state_dict": detector.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "kind": "snapshot_not_completion",
                    "training_resume_supported": False})
                last_snapshot = time.monotonic()
        if not seen:
            raise ValueError("Empty training epoch")
        train_seconds = time.monotonic() - tick
        dev_started = time.monotonic()
        scored = score_sources(encoder, detector, dev, device)
        dev_seconds = time.monotonic() - dev_started
        metrics = source_metrics(scored)
        if not np.isfinite(metrics["macro_eer"]):
            raise RuntimeError("Nonfinite source macro EER")
        improved = metrics["macro_eer"] < best
        if improved:
            best, best_epoch, stale = metrics["macro_eer"], epoch, 0
        else:
            stale += 1
        payload = {"config": config, "epoch": epoch, "fold": config["fold"], "seed": config["seed"],
                   "format": "frozen_encoder_detector_only_v1", "detector_state_dict": detector.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                   "sam_state": None if sam is None else sam.state_dict(), "source_dev": metrics,
                   "source_dev_scores": {d: torch.from_numpy(s.copy()) for d, (_, s) in scored.items()},
                   "optimizer_updates": updates, "bad_epochs": stale}
        for name in (("best.pt", "last.pt") if improved else ("last.pt",)):
            save_checkpoint(run/name, payload)
        with (run/"history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch, "source_dev": metrics, "best_epoch": best_epoch,
                "bad_epochs": stale, "train_seconds": train_seconds, "dev_seconds": dev_seconds,
                "train_losses": (sums/seen).tolist()})+"\n")
        print(f"EPOCH {epoch} macro_source_EER={metrics['macro_eer']:.8f} best={best_epoch} "
              f"patience={stale}/{recipe['patience']}", flush=True)
        if stale >= recipe["patience"]:
            break
    check_guards(config)
    return {"fold": config["fold"], "seed": config["seed"], "best_epoch": best_epoch, "final_epoch": epoch,
            "best_macro_source_dev_eer": best, "early_stopped": stale >= recipe["patience"],
            "wall_seconds": time.monotonic()-started, "optimizer_updates": updates, "target_access": False}


def finalize(encoder, detector, method, run, config, records, dev, device, completion):
    """Fresh detector reload, ALL canonical source scores, CSV round trip and calibration."""
    import csv

    check_guards(config)
    if json.loads((run/"config.json").read_text()) != config:
        raise RuntimeError("Saved run configuration changed; no completion marker")
    checkpoint = torch.load(run/"best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["format"] != "frozen_encoder_detector_only_v1":
        raise RuntimeError("Foreign selected checkpoint")
    detector.load_state_dict(checkpoint["detector_state_dict"], strict=True)
    scored = score_sources(encoder, detector, dev, device)
    if source_metrics(scored) != checkpoint["source_dev"] or any(
            not np.array_equal(s, checkpoint["source_dev_scores"][d].numpy()) for d, (_, s) in scored.items()):
        raise RuntimeError("Full selected-source reload differs; no completion marker")
    hashes, exported = {}, {}
    for domain, (labels, scores) in scored.items():
        if labels.tolist() != [y for _, y in records[domain]["dev"]]:
            raise RuntimeError("Canonical source labels/order differ")
        path = run/f"source_dev_{domain}.csv"
        write_score_csv(path, ((utterance_id(p, domain, config["data_root"]), domain, y, float(s))
                              for (p, y), s in zip(records[domain]["dev"], scores)))
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        # CSV writer must round-trip exactly, including calibration reduction order.
        y = np.array([int(r["label"]) for r in rows])
        s = np.array([float(r["raw_score"]) for r in rows], dtype=np.float32)
        if not np.array_equal(y, labels) or not np.array_equal(s, scores):
            raise RuntimeError("Source CSV round trip differs")
        exported[domain] = y, s
        hashes[path.name] = checkpoint_digest(path)
    calibration = calibrate_source_scores(scored)
    if calibrate_source_scores(exported) != calibration:
        raise RuntimeError("Exported source calibration differs")
    write_json(run/"source_calibration.json", calibration)
    if pretrained_identity(method) != config["pretrained"]:
        raise RuntimeError("Official pretrained files changed")
    check_guards(config)
    digest = checkpoint_digest(run/"best.pt")
    validation = {"selected_checkpoint_sha256": digest, "canonical_source_identity": config["source_identity"],
                  "all_source_scores_reloaded_bit_exact": True, "source_csv_sha256": hashes,
                  "exact_exported_calibration": True, "target_access": False}
    write_json(run/"source_validation.json", validation)
    write_json(run/"score_manifest.json", {"model": config["run_name"], "fold": config["fold"],
        "seed": config["seed"], "checkpoint_sha256": digest, "score_sha256": hashes,
        "source_validation_sha256": checkpoint_digest(run/"source_validation.json"), "target_access": False})
    completion.update(best_checkpoint_sha256=digest, config_sha256=checkpoint_digest(run/"config.json"),
        source_scores_sha256=hashes, source_calibration_sha256=checkpoint_digest(run/"source_calibration.json"),
        source_validation_sha256=checkpoint_digest(run/"source_validation.json"),
        score_manifest_sha256=checkpoint_digest(run/"score_manifest.json"),
        selected_checkpoint_reloaded_and_source_verified=True)
    write_json(run/"training_complete.json", completion)


def main(method, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--seed", choices=FINAL_SEEDS, type=int, default=1234)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--inspect", action="store_true", help="Print CPU detector count/recipe only; no weights/data/GPU")
    args = parser.parse_args(argv)
    if min(args.num_workers, args.eval_workers) < 0 or min(args.prefetch_factor, args.cpu_threads) < 1:
        parser.error("Invalid worker/thread/prefetch settings")
    torch.set_num_threads(args.cpu_threads)
    detector = DETECTORS[method]()
    if args.inspect:
        print(json.dumps({"method": method, "recipe": RECIPES[method], "detector_config": asdict(detector.cfg),
                          "trainable_parameters": sum(p.numel() for p in detector.parameters()),
                          "adaptation": True, "gpu_validation": "pending", "target_access": False}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use --inspect for CPU-only preparation")
    recipe = RECIPES[method]
    run = ROOT/"outputs"/recipe["family"]/args.fold/str(args.seed)
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent/f".{args.seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run.exists():
            raise FileExistsError(f"Preserving existing run: {run}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        set_seed(args.seed)
        pinned = pretrained_identity(method)
        records = source_records(args.fold, args.data_root)
        identity = source_identity(records, args.data_root)
        for parts in records.values():
            for rows in parts.values():
                if not all(Path(p).is_file() for p, _ in rows):
                    raise FileNotFoundError("Missing source audio; refusing partial membership")
        def loaders():
            train = make_train_loader([DetectorAudioDataset(parts["train"], method, training=True)
                for parts in records.values()], batch_size=recipe["batch_size"], seed=args.seed,
                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
            dev = {d: make_eval_loader(DetectorAudioDataset(parts["dev"], method, training=False),
                num_workers=args.eval_workers, prefetch_factor=args.prefetch_factor) for d, parts in records.items()}
            return train, dev
        encoder = FrozenSpeechEncoder(method).cuda()
        train, dev = loaders()
        detector = DETECTORS[method]().cuda()
        config = {**vars(args), **recipe, "data_root": str(args.data_root), "method": method,
            "run_name": recipe["family"], "detector_config": asdict(detector.cfg), "pretrained": pinned,
            "source_identity": identity, "sources": list(records), "target": FOLDS[args.fold]["target"],
            "source_sha256": source_guards(method), "target_access": False, "encoder_frozen": True,
            "encoder_execution": "eval_no_grad_FP32_no_waveform_normalization",
            "encoder_attention": "sdpa" if method == "lhcc" else "eager",
            "audio_loading": "16kHz_exact_seek_read; other_rates_full_resample_then_crop",
            "trainable_parameters": sum(p.numel() for p in detector.parameters()),
            "frozen_parameters": sum(p.numel() for p in encoder.parameters()), "precision": "float32",
            "selection": "strict_equal_domain_macro_source_dev_eer_ties_earlier",
            "sampler": "equal_domain_natural_classes_replacement_N_draws", "gradient_accumulation_steps": 1,
            "seed_plan": list(FINAL_SEEDS), "eval_batch_size": 1, "pin_memory": True,
            "rawboost": False, "specaugment": False, "mixup": False, "training_resume": False,
            "adaptation": "owner-approved paper-informed frozen-detector adaptation; not author-code reproduction",
            "assumptions_reference": f"scripts/{method}/README.md",
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
            "cuda_matmul_tf32": False, "cudnn_tf32": True,
            "fresh_initialization": "official frozen encoder plus fresh random detector; no probe/fold reuse"}
        run.mkdir(exist_ok=False)
        write_json(run/"config.json", config)
        check_guards(config)
        try:
            proof = preflight(encoder, detector, method, train, dev, "cuda")
            check_guards(config)
        except Exception as exc:
            write_json(run/"source_preflight_failed.json", {"error_type": type(exc).__name__,
                "error": str(exc), "target_access": False, "automatic_retry": False,
                "config_sha256": checkpoint_digest(run/"config.json")})
            raise
        proof["config_sha256"] = checkpoint_digest(run/"config.json")
        write_json(run/"source_preflight.json", proof)
        # Discard probe, optimizer, sampler/crop RNG and all probe loader workers.
        del detector, train, dev
        torch.cuda.empty_cache()
        set_seed(args.seed)
        detector = DETECTORS[method]().cuda()
        train, dev = loaders()
        torch.cuda.reset_peak_memory_stats()
        completion = fit(encoder, detector, method, train, dev, "cuda", run, config)
        # Reconstruct only the detector. Frozen official encoder bytes/grad invariants are checked separately.
        del detector
        torch.cuda.empty_cache()
        fresh = DETECTORS[method]().cuda()
        completion.update(peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        finalize(encoder, fresh, method, run, config, records, dev, "cuda", completion)
        print(f"COMPLETE {run}; selected full source scores/calibration verified; target unopened", flush=True)
