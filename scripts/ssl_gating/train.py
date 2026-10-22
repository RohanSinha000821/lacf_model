"""Train the repaired SSL-gating baseline on source train/dev partitions only."""

import argparse
import fcntl
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import compute_eer
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, validate_run_name
from audio_deepfake_detection.sota.ssl_gating import (
    FullAudioDataset, MODEL_CONFIG, SSLFeatureGating, UPSTREAM_COMMIT, TRAIN_MAX_SECONDS, cka_loss,
    make_eval_loader, make_train_loader,
)

BATCH_SIZE = 5
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = (0.9, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-name", type=validate_run_name, default="ssl_gating")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--gpu-memory-budget-gib", type=float,
                        help="Optional allocator budget for sharing a GPU; memory failure stops the run")
    args = parser.parse_args()
    if min(args.seed, args.num_workers, args.eval_workers) < 0 or args.prefetch_factor < 1:
        parser.error("Seed/workers must be nonnegative and prefetch positive")
    if args.gpu_memory_budget_gib is not None and (
            not math.isfinite(args.gpu_memory_budget_gib) or args.gpu_memory_budget_gib <= 0):
        parser.error("GPU memory budget must be finite and positive")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model):
    # optimizer_gamma in upstream YAML is unused by configure_optimizers().
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device))
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    for step, (waves, labels) in enumerate(loader, 1):
        waves, labels = waves.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, means = model(waves, return_block_means=True)
        ce, cka = criterion(logits, labels), cka_loss(means)
        loss = ce + cka
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; run stopped")
        loss.backward()
        optimizer.step()
        totals.add_(torch.stack((loss.detach(), ce.detach(), cka.detach())).double() * len(labels))
        count += len(labels)
        if step % 100 == 0:
            values = (totals / count).tolist()
            print(f"step={step}/{len(loader)} loss={values[0]:.6f} ce={values[1]:.6f} cka={values[2]:.6f}", flush=True)
    if not count:
        raise ValueError("Training loader produced no samples")
    return dict(zip(("total", "ce", "cka"), (totals / count).tolist()))


@torch.inference_mode()
def evaluate_eer(model, loader, device):
    model.eval()
    labels_all, scores_all = [], []
    for waves, labels in loader:
        scores_all.extend(model.spoof_score(model(waves.to(device, non_blocking=True))).float().cpu().tolist())
        labels_all.extend(labels.tolist())
    return compute_eer(labels_all, scores_all)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def code_provenance(root):
    paths = [Path(__file__), root / "src/audio_deepfake_detection/sota/ssl_gating.py",
             root / "src/audio_deepfake_detection/sota/ssl_gating_blocks.py",
             *[root / f"src/audio_deepfake_detection/{name}.py" for name in ("data", "protocol", "metrics")]]
    result = {"source_sha256": {str(p.relative_to(root)): checkpoint_digest(p) for p in paths}}
    for key, cmd in (("git_revision", ["git", "rev-parse", "HEAD"]),
                     ("git_status", ["git", "status", "--porcelain"])):
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else "unavailable"
    return result


def run(args, root, run_dir):
    set_seed(args.seed)
    fold = FOLDS[args.fold]
    print(f"SSL gating {args.fold}: sources={fold['sources']}; target remains unopened", flush=True)
    train_sets, dev_loaders, counts = [], {}, {}
    for domain in fold["sources"]:
        train_records = read_dataset(domain, "train", args.data_root)
        dev_records = read_dataset(domain, "dev", args.data_root)
        counts[domain] = {"train": len(train_records), "dev": len(dev_records)}
        print(f"{domain}: {counts[domain]}", flush=True)
        train_sets.append(FullAudioDataset(train_records, training=True))
        dev_loaders[domain] = make_eval_loader(FullAudioDataset(dev_records),
                                              num_workers=args.eval_workers, prefetch_factor=args.prefetch_factor)
    loader = make_train_loader(train_sets, batch_size=BATCH_SIZE, seed=args.seed,
                               num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    device = torch.device("cuda")
    model = SSLFeatureGating().to(device)
    optimizer = make_optimizer(model)
    config = {
        "run_name": args.run_name, "model_name": "SSLFeatureGating", "model_config": MODEL_CONFIG,
        "upstream_commit": UPSTREAM_COMMIT, "frontend_config": model.feature_extractor.config.to_dict(),
        "fold": args.fold, "sources": list(fold["sources"]), "target": fold["target"],
        "target_split": fold["target_split"], "seed": args.seed, "data_root": str(args.data_root),
        "dataset_counts": counts, "batch_size": BATCH_SIZE, "gradient_accumulation_steps": 1,
        "eval_batch_size": 1, "num_workers": args.num_workers, "eval_workers": args.eval_workers,
        "prefetch_factor": args.prefetch_factor, "precision": "float32", "optimizer": "Adam",
        "gpu_memory_budget_gib": args.gpu_memory_budget_gib,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "scheduler": "constant",
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "class_weights": CLASS_WEIGHTS,
        "loss": "weighted_ce_plus_mean_absolute_pairwise_cka", "cka_coefficient": 1.0,
        "selection": "strictly_lower_macro_source_dev_eer", "drop_last": False,
        "sampler": "equal_domain_natural_class_replacement", "rawboost": False,
        "input_policy": MODEL_CONFIG["input_policy"], "label_order": ["bona_fide", "spoof"],
        "train_input_policy": MODEL_CONFIG["train_input_policy"],
        "train_max_seconds": TRAIN_MAX_SECONDS, "crop_after_resampling": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "transformers_version": __import__("transformers").__version__,
        "torchaudio_version": __import__("torchaudio").__version__,
        "python_version": __import__("platform").python_version(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "cudnn_deterministic": True, "cudnn_benchmark": False,
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), **code_provenance(root),
    }
    write_json(run_dir / "config.json", config)
    best_eer, best_epoch, bad_epochs = float("inf"), 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        phase = time.monotonic()
        print(f"Epoch {epoch} training starts", flush=True)
        losses = train_epoch(model, loader, optimizer, device)
        train_seconds = time.monotonic() - phase
        print(f"Epoch {epoch} training took {train_seconds:.1f}s; source-dev scoring starts", flush=True)
        phase = time.monotonic()
        dev_eers = {domain: evaluate_eer(model, dev, device) for domain, dev in dev_loaders.items()}
        dev_seconds = time.monotonic() - phase
        macro = float(np.mean(list(dev_eers.values())))
        if not np.isfinite(macro):
            raise RuntimeError("Nonfinite development EER; no completion marker will be written")
        improved = macro < best_eer
        if improved:
            best_eer, best_epoch, bad_epochs = macro, epoch, 0
        else:
            bad_epochs += 1
        checkpoint = {
            "fold": args.fold, "epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "train_losses": losses, "dev_eers": dev_eers,
            "macro_dev_eer": macro, "best_macro_dev_eer": best_eer, "bad_epochs": bad_epochs,
            "optimizer_updates": epoch * len(loader), "train_seconds": train_seconds, "dev_seconds": dev_seconds,
        }
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = run_dir / (name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(run_dir / name)
        print(f"Epoch {epoch}: losses={losses}; source_eers={dev_eers}; macro={macro:.8f}; "
              f"best={best_eer:.8f}; patience={bad_epochs}/{PATIENCE}; dev_seconds={dev_seconds:.1f}", flush=True)
        if bad_epochs >= PATIENCE:
            break
    write_json(run_dir / "training_complete.json", {
        "fold": args.fold, "seed": args.seed, "best_epoch": best_epoch, "final_epoch": epoch,
        "best_macro_source_dev_eer": best_eer, "best_checkpoint_sha256": checkpoint_digest(run_dir / "best.pt"),
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    })
    print(f"Training complete: {run_dir}; target was not loaded", flush=True)


def main():
    args = parse_args()
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    run_dir = root / "outputs" / args.run_name / args.fold / str(args.seed)
    if any((run_dir / name).exists() for name in ("config.json", "best.pt", "last.pt", "training_complete.json", "metrics.json")):
        raise FileExistsError(f"Run already has artifacts; refusing to overwrite: {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final SSL-gating training")
    if args.gpu_memory_budget_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.gpu_memory_budget_gib * 1024 ** 3 / total
        if fraction > 1:
            raise ValueError("GPU memory budget exceeds this device's total capacity")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((run_dir / name).exists() for name in ("config.json", "best.pt", "last.pt", "training_complete.json", "metrics.json")):
            raise FileExistsError(f"Run already has artifacts; refusing to overwrite: {run_dir}")
        run(args, root, run_dir)


if __name__ == "__main__":
    main()
