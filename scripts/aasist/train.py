"""Train the fixed AASIST baseline using only a fold's source train/dev splits."""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import compute_eer
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, validate_run_name
from audio_deepfake_detection.sota.aasist import (
    AASIST, AASISTDataset, MODEL_CONFIG, UPSTREAM_COMMIT, make_eval_loader, make_train_loader,
)

BATCH_SIZE = 24
MAX_EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = (0.9, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-name", type=validate_run_name, default="aasist")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()
    if min(args.seed, args.num_workers, args.eval_workers) < 0 or args.prefetch_factor < 1:
        parser.error("Seed/workers must be nonnegative and prefetch factor positive")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model, *, steps_per_epoch):
    if steps_per_epoch < 1:
        raise ValueError("Training loader must have at least one step")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), amsgrad=False)
    # Upstream cosine is stepped per optimizer update, not once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS * steps_per_epoch, eta_min=MIN_LEARNING_RATE)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device))
    loss_sum, count = 0.0, 0
    for step, (waveforms, labels) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(waveforms), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; run stopped")
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * len(labels)
        count += len(labels)
        if step % 100 == 0:
            print(f"step={step:05d}/{len(loader):05d} loss={loss_sum / count:.6f}", flush=True)
    if count == 0:
        raise ValueError("Training loader produced no samples")
    return loss_sum / count


@torch.inference_mode()
def evaluate_eer(model, loader, device):
    model.eval()
    labels_all, scores_all = [], []
    for waveforms, labels in loader:
        logits = model(waveforms.to(device, non_blocking=True))
        scores_all.extend(model.spoof_score(logits).float().cpu().tolist())
        labels_all.extend(labels.tolist())
    return compute_eer(labels_all, scores_all)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_provenance(project_root):
    paths = [Path(__file__), project_root / "src/audio_deepfake_detection/sota/aasist.py",
             project_root / "src/audio_deepfake_detection/sota/aasist_arch.py",
             project_root / "src/audio_deepfake_detection/data.py",
             project_root / "src/audio_deepfake_detection/protocol.py",
             project_root / "src/audio_deepfake_detection/metrics.py"]
    result = {"source_sha256": {str(path.relative_to(project_root)): checkpoint_digest(path)
                                for path in paths if path.is_file()}}
    for key, command in (("git_revision", ["git", "rev-parse", "HEAD"]),
                         ("git_status", ["git", "status", "--porcelain"])):
        completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True, check=False)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return result


def main():
    args = parse_args()
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run_dir = project_root / "outputs" / args.run_name / args.fold / str(args.seed)
    if any((run_dir / name).exists() for name in ("best.pt", "last.pt", "training_complete.json", "metrics.json")):
        raise FileExistsError(f"Run already has results; refusing to overwrite: {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final AASIST training")
    set_seed(args.seed)
    fold = FOLDS[args.fold]
    print(f"AASIST {args.fold}: sources={fold['sources']}; target={fold['target']} remains unopened", flush=True)
    train_sets, dev_loaders, counts = [], {}, {}
    for domain in fold["sources"]:
        train_records = read_dataset(domain, "train", args.data_root)
        dev_records = read_dataset(domain, "dev", args.data_root)
        counts[domain] = {"train": len(train_records), "dev": len(dev_records)}
        print(f"{domain}: {counts[domain]}", flush=True)
        train_sets.append(AASISTDataset(train_records, training=True))
        dev_loaders[domain] = make_eval_loader(AASISTDataset(dev_records, training=False),
                                              num_workers=args.eval_workers, prefetch_factor=args.prefetch_factor)
    train_loader = make_train_loader(train_sets, batch_size=BATCH_SIZE, seed=args.seed,
                                     num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    device = torch.device("cuda")
    model = AASIST().to(device)
    optimizer, scheduler = make_optimizer(model, steps_per_epoch=len(train_loader))
    config = {
        "run_name": args.run_name, "model_name": "AASIST", "model_config": MODEL_CONFIG,
        "upstream_commit": UPSTREAM_COMMIT, "fold": args.fold, "sources": list(fold["sources"]),
        "target": fold["target"], "target_split": fold["target_split"], "seed": args.seed,
        "data_root": str(args.data_root), "batch_size": BATCH_SIZE, "eval_batch_size": 1,
        "gradient_accumulation_steps": 1, "num_workers": args.num_workers,
        "eval_workers": args.eval_workers, "prefetch_factor": args.prefetch_factor,
        "learning_rate": LEARNING_RATE, "minimum_learning_rate": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "class_weights": CLASS_WEIGHTS,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "precision": "float32",
        "optimizer": "Adam", "scheduler": "cosine_per_optimizer_update",
        "scheduler_total_steps": MAX_EPOCHS * len(train_loader), "drop_last": False,
        "sampler": "equal_domain_natural_class_replacement", "dataset_counts": counts,
        "train_crop": "random_64600_after_16k_resampling", "eval_crop": "first_64600_repeat_if_short",
        "freq_aug": False, "rawboost": False, "swa": False,
        "selection": "strictly_lower_macro_source_dev_eer", "label_order": ["bona_fide", "spoof"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "python_version": __import__("platform").python_version(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "torchaudio_version": __import__("torchaudio").__version__,
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "deterministic_algorithms_enforced": False,
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        **code_provenance(project_root),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    print(f"batch={BATCH_SIZE}; steps/epoch={len(train_loader)}; precision=float32", flush=True)
    best_eer, best_epoch, bad_epochs = float("inf"), 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.monotonic()
        print(f"Epoch {epoch} training starts", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_seconds = time.monotonic() - epoch_start
        dev_start = time.monotonic()
        dev_eers = {domain: evaluate_eer(model, loader, device) for domain, loader in dev_loaders.items()}
        macro = float(np.mean(list(dev_eers.values())))
        if not np.isfinite(macro):
            raise RuntimeError("Nonfinite development EER; no completion marker will be written")
        improved = macro < best_eer
        if improved:
            best_eer, best_epoch, bad_epochs = macro, epoch, 0
        else:
            bad_epochs += 1
        checkpoint = {
            "fold": args.fold, "epoch": epoch, "config": config,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "train_loss": train_loss,
            "dev_eers": dev_eers, "macro_dev_eer": macro, "best_macro_dev_eer": best_eer,
            "bad_epochs": bad_epochs, "optimizer_updates": epoch * len(train_loader),
            "train_seconds": train_seconds, "dev_seconds": time.monotonic() - dev_start,
        }
        # Temporary files avoid exposing a partially written checkpoint.
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = run_dir / (name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(run_dir / name)
        print(f"Epoch {epoch}: loss={train_loss:.6f}; dev_eers={dev_eers}; "
              f"macro={macro * 100:.6f}%; best={best_eer * 100:.6f}%; "
              f"early stopping={bad_epochs}/{PATIENCE}", flush=True)
        if bad_epochs >= PATIENCE:
            print("Early stopping reached", flush=True)
            break
    write_json(run_dir / "training_complete.json", {
        "fold": args.fold, "seed": args.seed, "best_epoch": best_epoch, "final_epoch": epoch,
        "best_macro_source_dev_eer": best_eer, "best_checkpoint": str(run_dir / "best.pt"),
        "best_checkpoint_sha256": checkpoint_digest(run_dir / "best.pt"),
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    })
    print(f"Training complete: {run_dir}; held-out target was not loaded", flush=True)


if __name__ == "__main__":
    main()
