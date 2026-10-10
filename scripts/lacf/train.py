"""Train primary frozen LACF on one fold's official source train/dev splits only.

No target exporter, ablation queue, FT4, resume or automatic weight download.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torchaudio
import transformers

from audio_deepfake_detection.lacf.data import LACFDataset, make_loader
from audio_deepfake_detection.lacf.cache import frozen_development
from audio_deepfake_detection.lacf.pretrained import load_local_model
from audio_deepfake_detection.lacf.model import PRIMARY_TRAINABLE_PARAMETERS
from audio_deepfake_detection.lacf.training import make_optimizer, source_eer, train_epoch
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, final_seeds_for, read_dataset, verify_source_macro,
)

BATCH_SIZE, ACCUMULATION_STEPS, MAX_EPOCHS, PATIENCE = 4, 8, 30, 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, choices=final_seeds_for("lacf"), default=1234)
    parser.add_argument("--segment-seconds", type=int, choices=(4, 10), default=10)
    parser.add_argument("--output-root", type=Path, help="Fresh output family directory; existing runs are never overwritten")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--confirm-cache-verified", action="store_true")
    args = parser.parse_args()
    if (min(args.num_workers, args.eval_workers) < 0
            or min(args.prefetch_factor, args.batch_size, args.gradient_accumulation_steps) < 1):
        parser.error("Workers must be nonnegative; batch, accumulation and prefetch must be positive")
    return args


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_provenance(root):
    paths = [Path(__file__), root / "scripts/lacf/train_folds.sh",
             *sorted((root / "src/audio_deepfake_detection/lacf").glob("*.py")),
             *(root / f"src/audio_deepfake_detection/{name}.py" for name in ("data", "metrics", "protocol"))]
    result = {"source_sha256": {str(path.relative_to(root)): checkpoint_digest(path) for path in paths}}
    for key, command in (("git_revision", ["git", "rev-parse", "HEAD"]),
                         ("git_status", ["git", "status", "--porcelain"])):
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        result[key] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return result


def development_cache_identity(config):
    """Bind fixed source-dev features to the same processors, encoders and code."""
    return {"segment_seconds": config["segment_seconds"], "input_policy": config["input_policy"],
            "revisions": {key: config[key] for key in ("wavlm_revision", "clap_revision")},
            "enabled_encoders": [config["components"][key] for key in ("wavlm_enabled", "clap_audio_enabled")],
            "encoding_code": {key: value for key, value in config["source_sha256"].items()
                              if key.endswith(("model.py", "pretrained.py", "precision.py", "data.py"))}}


def main():
    args = parse_args()
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    family = "lacf4s" if args.segment_seconds == 4 else "lacf"
    output_root = args.output_root or root / "outputs" / family
    run_dir = output_root / args.fold / str(args.seed)
    # Preserve even partial/unrelated work; no implicit resume or replacement.
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if not args.confirm_cache_verified:
        raise ValueError("Verified transfer/content integrity must be acknowledged before training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual LACF training; use synthetic tests for CPU checks")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 support is required; no precision fallback")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    # Load cached encoders before creating run files. Missing weights fail closed.
    model, collator, revisions = load_local_model(segment_seconds=args.segment_seconds)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if (sum(p.numel() for p in trainable.values()) != PRIMARY_TRAINABLE_PARAMETERS
            or any(not name.startswith(("adapter.", "classifier.")) for name in trainable)):
        raise ValueError("Primary LACF must train only the 661,479 adapter/classifier parameters")
    fold = FOLDS[args.fold]
    train_sets, dev_loaders, counts, development_records = [], {}, {}, {}
    for domain in fold["sources"]:
        train_records = read_dataset(domain, "train", args.data_root)
        dev_records = read_dataset(domain, "dev", args.data_root)
        development_records[domain] = dev_records
        counts[domain] = {"train": len(train_records), "dev": len(dev_records)}
        train_sets.append(LACFDataset(train_records, training=True, segment_seconds=args.segment_seconds))
        dev_loaders[domain] = make_loader([LACFDataset(dev_records, training=False, segment_seconds=args.segment_seconds)], collator,
            training=False, seed=args.seed, batch_size=1, num_workers=args.eval_workers,
            prefetch_factor=args.prefetch_factor)
    train_loader = make_loader(train_sets, collator, training=True, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = make_optimizer(model)
    config = {
        "model_name": "LACF-Frozen", "run_name": family, "fold": args.fold, "seed": args.seed,
        "segment_seconds": args.segment_seconds, "output_root": str(output_root),
        "final_seed_plan": list(final_seeds_for("lacf")), "seed_policy": "single_seed_compute_budget",
        "sources": list(fold["sources"]),
        "target": fold["target"], "target_split": fold["target_split"], "data_root": str(args.data_root),
        "dataset_counts": counts, "components": model.configuration(), "input_policy": collator.configuration(),
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps, "eval_batch_size": 1,
        "num_workers": args.num_workers, "eval_workers": args.eval_workers, "prefetch_factor": args.prefetch_factor,
        "sampler": "equal_domain_equal_class_uniform_utterance_replacement", "drop_last": False,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8, "scheduler": "constant",
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "precision": "bfloat16_mixed",
        "parameter_dtype": "float32", "sensitive_dtype": "float32", "gradient_scaler": False,
        "automatic_learning_rate_scaling": False,
        "selection": "strictly_lower_macro_source_dev_eer", "label_order": ["bona_fide", "spoof"],
        "score": "scalar_spoof_logit", "augmentation": "none",
        "accumulation_loss": "mean_of_actual_microbatch_losses_including_final_incomplete_group",
        "cache_verified_by_operator": args.confirm_cache_verified,
        "cache_copy_checksum_status": "verified_by_operator",
        "development_execution": "cached_frozen_audio_batch1_current_backend_batch1",
        "development_cache_scope": "source_only_centered_crops_no_trainable_features_or_scores",
        "nnpack_enabled": False,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "torchaudio_version": torchaudio.__version__,
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "hf_home": os.environ.get("HF_HOME", ""),
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms_enforced": False,
        "prototype_sha256": hashlib.sha256(model.prototypes.cpu().numpy().tobytes()).hexdigest(),
        "recipe_status": "starting_recipe_requires_source_only_validation_before_final_target_access",
        **revisions, **code_provenance(root),
    }
    validation_path = os.environ.get("LACF_SOURCE_VALIDATION_RECORD")
    if validation_path:
        validation = json.loads(Path(validation_path).read_text())
        recipe_keys = ("seed", "segment_seconds", "batch_size", "gradient_accumulation_steps", "num_workers",
                       "eval_workers", "prefetch_factor", "precision", "trainable_parameters",
                       "learning_rate", "weight_decay", "scheduler", "max_epochs", "patience")
        if (validation.get("status") != "passed"
                or any(validation.get("recipe", {}).get(key) != config[key] for key in recipe_keys)
                or validation.get("source_sha256") != config["source_sha256"]
                or validation.get("revisions") != {key: config[key] for key in ("wavlm_revision", "clap_revision")}):
            raise ValueError("Frozen source-validation recipe/code/checkpoints do not match this run")
        config.update(source_validation_record=str(Path(validation_path).resolve()),
                      source_validation_sha256=checkpoint_digest(Path(validation_path)),
                      source_validation_scope=validation.get("source_validation_scope", "full_source_dev_and_reload"),
                      recipe_status="source_validated_frozen",
                      cache_copy_checksum_status=validation["copy_checksum_status"],
                      checksum_evidence_source=validation["checksum_evidence_source"],
                      checksum_evidence_sha256=validation["checksum_evidence_sha256"])
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    log = logging.getLogger(f"lacf.{args.fold}.{args.seed}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handlers = [logging.FileHandler(run_dir / "training.log"), logging.StreamHandler()]
    for handler in handlers:
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        formatter.converter = lambda timestamp: datetime.fromtimestamp(timestamp, ZoneInfo("Europe/Berlin")).timetuple()
        handler.setFormatter(formatter)
        log.addHandler(handler)
    best, best_epoch, bad_epochs, updates = float("inf"), 0, 0, 0
    try:
        log.info("%s | seed %s | %ss | batch %s x accumulation %s | trainable %s | times Europe/Berlin",
                 args.fold.upper(), args.seed, args.segment_seconds, args.batch_size,
                 args.gradient_accumulation_steps, config["trainable_parameters"])
        log.info("sources=%s | target=%s remains unopened", fold["sources"], fold["target"])
        cache_identity = development_cache_identity(config)
        config["development_feature_caches"] = {}
        for epoch in range(1, MAX_EPOCHS + 1):
            log.info("epoch %s/%s | training starts | %s batches", epoch, MAX_EPOCHS, len(train_loader))
            losses, steps = train_epoch(model, train_loader, optimizer, device,
                                       accumulation_steps=args.gradient_accumulation_steps, logger=log,
                                       phase=f"epoch {epoch} train")
            updates += steps
            log.info("epoch %s | training finished | loss %.5f | source development starts", epoch, losses["total"])
            if epoch == 1:
                # Build fixed features during first development, after useful training.
                for domain in fold["sources"]:
                    log.info("dev cache %s: encode fixed crops once, then reuse frozen features", domain)
                    dev_loaders[domain] = frozen_development(model, dev_loaders[domain], development_records[domain], device,
                        identity={**cache_identity, "domain": domain}, directory=root / "outputs/lacf_frozen_dev_features",
                        logger=log, phase="cache " + domain)
                    cache = dev_loaders[domain]
                    config["development_feature_caches"][domain] = {"path": str(cache.path), "sha256": cache.sha256}
                save_json(run_dir / "config.json", config)
            dev_eers = {}
            for domain, loader in dev_loaders.items():
                dev_eers[domain] = source_eer(model, loader, device, logger=log, phase=f"epoch {epoch} dev {domain}")
                log.info("epoch %s | %s source-dev EER %.3f%%", epoch, domain, 100 * dev_eers[domain])
            macro = float(np.mean(list(dev_eers.values())))
            verify_source_macro(macro, macro)
            improved = macro < best
            if improved:
                best, best_epoch, bad_epochs = macro, epoch, 0
            else:
                bad_epochs += 1
            checkpoint = {"fold": args.fold, "epoch": epoch, "config": config,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": losses, "dev_eers": dev_eers, "macro_dev_eer": macro,
                "best_macro_dev_eer": best, "bad_epochs": bad_epochs, "optimizer_updates": updates}
            for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
                temporary = run_dir / (name + ".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(run_dir / name)
            log.info("epoch=%s losses=%s dev_eers=%s macro=%s best=%s bad_epochs=%s/%s",
                     epoch, losses, dev_eers, macro, best, bad_epochs, PATIENCE)
            if bad_epochs >= PATIENCE:
                break
        selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        for name, value in model.state_dict().items():
            if name.startswith(("wavlm.", "clap.", "prototypes")) and not torch.equal(value.cpu(), selected["model_state_dict"][name]):
                raise ValueError("Frozen encoder/prototype changed; development cache cannot verify this checkpoint")
        model.load_state_dict(selected["model_state_dict"])
        log.info("selected checkpoint reload verification starts")
        reloaded = {domain: source_eer(model, loader, device, logger=log, phase="reload " + domain)
                    for domain, loader in dev_loaders.items()}
        verify_source_macro(float(np.mean(list(reloaded.values()))), best)
        for domain, value in reloaded.items():
            verify_source_macro(value, selected["dev_eers"][domain])
        digest = checkpoint_digest(run_dir / "best.pt")
        save_json(run_dir / "config.json", {**config, "selected_epoch": best_epoch,
                  "selected_macro_source_dev_eer": best, "selected_checkpoint_sha256": digest})
        save_json(run_dir / "training_complete.json", {
            "fold": args.fold, "seed": args.seed, "best_epoch": best_epoch, "final_epoch": epoch,
            "best_macro_source_dev_eer": best, "reloaded_dev_eers": reloaded,
            "best_checkpoint": str(run_dir / "best.pt"),
            "best_checkpoint_sha256": digest, "optimizer_updates": updates,
            "selected_checkpoint_reloaded_and_source_verified": True})
        log.info("Training complete; selected source checkpoint reloaded; held-out target never loaded")
    except Exception:
        log.exception("Run failed; partial work preserved")
        raise
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
