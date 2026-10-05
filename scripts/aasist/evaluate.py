"""Export fixed-length AASIST scores and apply the shared LODO evaluator."""

import argparse
import json
import os
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, read_dataset,
    save_score_manifest, utterance_id, validate_run_name, verify_completed_run,
    verify_score_digest, verify_source_macro,
)
from audio_deepfake_detection.sota.aasist import AASIST, AASISTDataset, MODEL_CONFIG, UPSTREAM_COMMIT, make_eval_loader


def verify_checkpoint(checkpoint, completion, fold_name, seed, run_name):
    config = checkpoint["config"]
    fold = FOLDS[fold_name]
    if (checkpoint["fold"] != fold_name or config["fold"] != fold_name
            or config["seed"] != seed or config["run_name"] != run_name
            or config["model_name"] != "AASIST" or config["model_config"] != MODEL_CONFIG
            or config["upstream_commit"] != UPSTREAM_COMMIT
            or config["precision"] != "float32"
            or config["eval_crop"] != "first_64600_repeat_if_short"
            or tuple(config["sources"]) != fold["sources"]
            or config["target"] != fold["target"] or config["target_split"] != fold["target_split"]
            or checkpoint["epoch"] != completion["best_epoch"]):
        raise ValueError("Checkpoint identity, recipe or selected epoch does not match the AASIST run")
    verify_source_macro(checkpoint["macro_dev_eer"], completion["best_macro_source_dev_eer"])


@torch.inference_mode()
def score_rows(model, loader, records, dataset_name, data_root, device):
    model.eval()
    offset = 0
    for waveforms, labels in loader:
        scores = model.spoof_score(model(waveforms.to(device, non_blocking=True))).float().cpu().tolist()
        batch_records = records[offset:offset + len(scores)]
        if len(batch_records) != len(scores) or len(labels) != len(scores):
            raise RuntimeError("Scoring output count differs from input records")
        for (path, expected_label), label, score in zip(batch_records, labels.tolist(), scores):
            if label != expected_label:
                raise RuntimeError("Evaluation loader changed record-label order")
            yield utterance_id(path, dataset_name, data_root), dataset_name, label, score
        offset += len(scores)
        if offset % 5000 == 0:
            print(f"{dataset_name}: {offset:,}/{len(records):,} scored", flush=True)
    if offset != len(records):
        raise RuntimeError(f"Scored {offset} of {len(records)} records")


def score_split(model, dataset_name, split, data_root, output, device, *, num_workers, prefetch_factor, manifest):
    records = read_dataset(dataset_name, split, data_root)
    expected = canonical_score_labels(dataset_name, split, data_root, records=records)
    if manifest["splits"].get(output.name) != {"dataset": dataset_name, "split": split}:
        raise ValueError("Score manifest does not match requested split")
    if output.exists():
        verify_score_digest(output, manifest)
        return read_score_csv(output, dataset_name, expected_labels=expected)
    if output.name in manifest["score_sha256"]:
        raise ValueError(f"Previously registered score file is missing: {output}")
    loader = make_eval_loader(AASISTDataset(records, training=False),
                              num_workers=num_workers, prefetch_factor=prefetch_factor)
    write_score_csv(output, score_rows(model, loader, records, dataset_name, data_root, device))
    result = read_score_csv(output, dataset_name, expected_labels=expected)
    manifest["score_sha256"][output.name] = checkpoint_digest(output)
    save_score_manifest(output.parent, manifest)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-name", type=validate_run_name, default="aasist")
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/salt/datasets/audio-deepfake"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confirm-protocol-frozen", action="store_true",
                        help="Explicitly confirm the comparison-wide freeze BEFORE any target results")
    args = parser.parse_args()
    if min(args.seed, args.num_workers, args.bootstrap_resamples) < 0 or args.prefetch_factor < 1:
        parser.error("Seed/workers/resamples must be nonnegative and prefetch positive")
    if not args.confirm_protocol_frozen:
        parser.error("Target evaluation waits for the comparison-wide freeze; confirmation is required")
    return args


def main():
    args = parse_args()
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run_dir = project_root / "outputs" / args.run_name / args.fold / str(args.seed)
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Metrics already exist; refusing to overwrite: {run_dir}")
    completion, digest = verify_completed_run(run_dir, args.fold, args.seed)
    if completion.get("best_checkpoint_sha256") != digest:
        raise ValueError("Best checkpoint bytes differ from the completion record")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    verify_checkpoint(checkpoint, completion, args.fold, args.seed, args.run_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AASIST scoring")
    device = torch.device("cuda")
    model = AASIST()
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device).eval()
    manifest = load_score_manifest(run_dir, args.fold, args.seed, digest, create=True)
    fold = FOLDS[args.fold]
    source_dev = {}
    for domain in fold["sources"]:
        source_dev[domain] = score_split(model, domain, "dev", args.source_data_root,
                                        run_dir / f"source_dev_{domain}.csv", device,
                                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                        manifest=manifest)
    calibration = calibrate_source_scores(source_dev)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    print("Source-only calibration verified; opening the held-out target", flush=True)
    target_labels, target_scores = score_split(model, fold["target"], fold["target_split"],
                                              args.target_data_root, run_dir / "target_scores.csv", device,
                                              num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                              manifest=manifest)
    report = evaluate_transfer(source_dev, target_labels, target_scores,
                               bootstrap_resamples=args.bootstrap_resamples)
    report.update({
        "model": args.run_name, "fold": args.fold, "seed": args.seed, "sources": list(fold["sources"]),
        "target_domain": fold["target"], "target_split": fold["target_split"],
        "checkpoint_sha256": digest, "checkpoint_epoch": completion["best_epoch"],
        "score_sha256": dict(manifest["score_sha256"]), "evaluator_sha256": checkpoint_digest(Path(__file__)),
        "source_data_root": str(args.source_data_root), "target_data_root": str(args.target_data_root),
        "comparison_wide_freeze_confirmed_by_operator": True,
        "precision": "float32", "input_policy": "16kHz_first_64600_repeat_if_short",
    })
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "metrics.json")
    print(f"Target EER={report['target']['eer'] * 100:.4f}%; AUROC={report['target']['auroc']:.6f}")


if __name__ == "__main__":
    main()
