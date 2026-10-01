from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import compute_eer
from audio_deepfake_detection.protocol import DISPLAY_NAMES, FOLDS, read_dataset

from audio_deepfake_detection.sota.wavlm_wa import (
    WavLMDataset,
    WavLMWA,
    make_wavlm_eval_loader,
    make_wavlm_train_loader,
)


# Training configuration

SEED = 1234
MODEL_NAME = "microsoft/wavlm-base"

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True

ENCODER_LR = 2e-5
BACKEND_LR = 5e-3
LR_GAMMA = 0.95

CLASS_WEIGHTS = (9.0, 1.0)

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5


# =========================================================
# Command line
# =========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train WavLM-WA under the LODO protocol"
    )

    parser.add_argument(
        "--fold",
        required=True,
        choices=(
            "f1",
            "f2",
            "f3",
            "f4",
        ),
        help="LODO fold to train",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/drive/audio-deepfake-cache"),
        help="Source train/dev dataset root (use the verified local cache)",
    )

    parser.add_argument("--seed", type=int, default=SEED, help="Training seed and output-run name")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Data-loader workers for training and evaluation (0 disables multiprocessing)",
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=PREFETCH_FACTOR,
        help="Batches to prefetch per worker (ignored when --num-workers is 0)",
    )

    args = parser.parse_args()

    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    return args


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =========================================================
# Development evaluation
# =========================================================

@torch.inference_mode()
def evaluate_eer(
    model: WavLMWA,
    loader,
    device: torch.device,
) -> float:

    model.eval()

    labels_all: list[int] = []
    scores_all: list[float] = []

    for (
        waveforms,
        attention_masks,
        labels,
    ) in loader:

        waveforms = waveforms.to(
            device,
            non_blocking=True,
        )

        attention_masks = (
            attention_masks.to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            logits = model(
                waveforms,
                attention_masks,
            )

        scores = model.spoof_score(
            logits
        )

        labels_all.extend(
            labels.tolist()
        )

        scores_all.extend(
            scores.float()
            .cpu()
            .tolist()
        )

    return compute_eer(
        labels_all,
        scores_all,
    )


# =========================================================
# Main training
# =========================================================

def main() -> None:

    args = parse_args()

    fold_name = args.fold

    fold = FOLDS[
        fold_name
    ]

    source_names = fold[
        "sources"
    ]

    target_name = fold[
        "target"
    ]

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required"
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
    )

    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

    output_dir = (
        project_root
        / "outputs"
        / "wavlm"
        / fold_name
        / str(args.seed)
    )

    if (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists():
        raise FileExistsError(f"Run already has checkpoints; refusing to overwrite: {output_dir}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "best.pt"
    )

    last_path = (
        output_dir
        / "last.pt"
    )

    # =====================================================
    # Print experiment definition
    # =====================================================

    print()
    print("=" * 60)

    print(
        f"WavLM-WA {fold_name.upper()}"
    )

    print("=" * 60)

    print(
        "Source domains:"
    )

    for name in source_names:

        print(
            " ",
            DISPLAY_NAMES[name],
        )

    print(
        "Held-out target:",
        DISPLAY_NAMES[target_name],
    )

    print(
        "Target data will NOT be loaded."
    )

    print("=" * 60)
    print()

    # =====================================================
    # Read source train/dev records
    # =====================================================

    train_records = {}

    dev_records = {}

    for dataset_name in source_names:

        print(
            "Reading",
            DISPLAY_NAMES[
                dataset_name
            ],
        )

        train_records[
            dataset_name
        ] = read_dataset(
            dataset_name,
            "train",
            args.data_root,
        )

        dev_records[
            dataset_name
        ] = read_dataset(
            dataset_name,
            "dev",
            args.data_root,
        )

        print(
            "  train:",
            len(
                train_records[
                    dataset_name
                ]
            ),
        )

        print(
            "  dev:",
            len(
                dev_records[
                    dataset_name
                ]
            ),
        )

    total_train = sum(
        len(records)
        for records
        in train_records.values()
    )

    print()
    print(
        "Total source train:",
        total_train,
    )

    # =====================================================
    # Training loader
    # =====================================================

    train_datasets = [
        WavLMDataset(
            train_records[name],
            training=True,
        )
        for name in source_names
    ]

    train_loader = (
        make_wavlm_train_loader(
            train_datasets,
            batch_size=BATCH_SIZE,
            seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=PIN_MEMORY,
        )
    )

    # =====================================================
    # Development loaders
    # =====================================================

    dev_loaders = {}

    for dataset_name in source_names:

        dataset = WavLMDataset(
            dev_records[
                dataset_name
            ],
            training=False,
        )

        dev_loaders[
            dataset_name
        ] = make_wavlm_eval_loader(
            dataset,
            batch_size=EVAL_BATCH_SIZE,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=PIN_MEMORY,
        )

    # =====================================================
    # Model
    # =====================================================

    print()
    print(
        "Loading WavLM-WA..."
    )

    model = WavLMWA(
        model_name=MODEL_NAME,
    ).to(
        device
    )

    # =====================================================
    # Optimizer
    # =====================================================

    backend_parameters = [
        model.layer_weights,
        *model.classifier.parameters(),
    ]

    optimizer = torch.optim.Adam(
        [
            {
                "params":
                    model.wavlm.parameters(),
                "lr":
                    ENCODER_LR,
            },

            {
                "params":
                    backend_parameters,
                "lr":
                    BACKEND_LR,
            },
        ]
    )

    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=LR_GAMMA,
        )
    )

    # =====================================================
    # Loss
    # =====================================================

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights,
        )
    )

    # =====================================================
    # Training state
    # =====================================================

    best_macro_eer = float(
        "inf"
    )

    best_epoch = 0

    bad_epochs = 0

    print()
    print(
        "Training starts"
    )

    print(
        "batch size:",
        BATCH_SIZE,
    )

    print(
        "steps per epoch:",
        len(train_loader),
    )

    print()

    # =====================================================
    # Epoch loop
    # =====================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        for (
            step,
            (
                waveforms,
                attention_masks,
                labels,
            ),
        ) in enumerate(
            train_loader,
            start=1,
        ):

            waveforms = waveforms.to(
                device,
                non_blocking=True,
            )

            attention_masks = (
                attention_masks.to(
                    device,
                    non_blocking=True,
                )
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                logits = model(
                    waveforms,
                    attention_masks,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.detach().item()
            )

            if step % 100 == 0:

                print(
                    f"epoch={epoch:03d} "
                    f"step={step:05d}/"
                    f"{len(train_loader):05d} "
                    f"loss="
                    f"{running_loss / step:.6f}"
                )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        print()
        print(
            f"Epoch {epoch} "
            f"training complete"
        )

        print(
            f"train loss: "
            f"{train_loss:.6f}"
        )

        # =================================================
        # Source-development evaluation
        # =================================================

        dev_eers = {}

        for dataset_name in source_names:

            eer = evaluate_eer(
                model,
                dev_loaders[
                    dataset_name
                ],
                device,
            )

            dev_eers[
                dataset_name
            ] = eer

            print(
                f"{DISPLAY_NAMES[dataset_name]} "
                f"dev EER: "
                f"{eer * 100:.6f}%"
            )

        macro_eer = (
            sum(
                dev_eers.values()
            )
            / len(dev_eers)
        )

        print(
            "macro source-dev EER: "
            f"{macro_eer * 100:.6f}%"
        )

        # =================================================
        # Best checkpoint / early stopping
        # =================================================

        improved = (
            macro_eer
            < best_macro_eer
        )

        if improved:

            best_macro_eer = (
                macro_eer
            )

            best_epoch = epoch

            bad_epochs = 0

        else:

            bad_epochs += 1

        # Scheduler changes LR for the next epoch.
        scheduler.step()

        # =================================================
        # Save checkpoint
        # =================================================

        checkpoint = {
            "fold":
                fold_name,

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_loss":
                train_loss,

            "dev_eers":
                dev_eers,

            "macro_dev_eer":
                macro_eer,

            "best_macro_dev_eer":
                best_macro_eer,

            "bad_epochs":
                bad_epochs,

            "config": {
                "model_name":
                    MODEL_NAME,

                "fold":
                    fold_name,

                "sources":
                    source_names,

                "target":
                    target_name,

                "data_root":
                    str(args.data_root),

                "output_dir":
                    str(output_dir),

                "seed":
                    args.seed,

                "batch_size":
                    BATCH_SIZE,

                "eval_batch_size":
                    EVAL_BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "num_workers":
                    args.num_workers,

                "prefetch_factor":
                    args.prefetch_factor,

                "pin_memory":
                    PIN_MEMORY,

                "encoder_lr":
                    ENCODER_LR,

                "backend_lr":
                    BACKEND_LR,

                "gamma":
                    LR_GAMMA,

                "class_weights":
                    CLASS_WEIGHTS,

                "max_epochs":
                    MAX_EPOCHS,

                "patience":
                    EARLY_STOPPING_PATIENCE,

                "bf16":
                    True,
            },
        }

        torch.save(
            checkpoint,
            last_path,
        )

        if improved:

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "new best checkpoint saved"
            )

        print(
            "best macro EER: "
            f"{best_macro_eer * 100:.6f}%"
        )

        print(
            "early stopping: "
            f"{bad_epochs}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        print(
            "next encoder LR:",
            optimizer.param_groups[
                0
            ]["lr"],
        )

        print(
            "next backend LR:",
            optimizer.param_groups[
                1
            ]["lr"],
        )

        print()

        if (
            bad_epochs
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping reached."
            )

            break

    # =====================================================
    # Finished
    # =====================================================

    print()
    print("=" * 60)

    print(
        f"{fold_name.upper()} "
        f"training finished"
    )

    print(
        "best macro source-dev EER:",
        f"{best_macro_eer * 100:.6f}%",
    )

    print(
        "best checkpoint:",
        best_path,
    )

    (output_dir / "config.json").write_text(
        json.dumps(checkpoint["config"], indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "training_complete.json").write_text(
        json.dumps({
            "fold": fold_name,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "final_epoch": epoch,
            "best_macro_source_dev_eer": best_macro_eer,
            "best_checkpoint": str(best_path),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
