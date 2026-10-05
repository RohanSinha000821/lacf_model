"""Run the common LODO metrics on score CSVs from any detector.

Expected files in --run-dir: source_dev_<dataset>.csv for each source domain,
and target_scores.csv. Each file has utterance_id,dataset,label,raw_score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, read_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest,
    verify_completed_run, verify_score_digest, verify_source_macro,
)


def evaluate_run(
    run_dir: Path, fold_name: str, *, seed: int, source_data_root: Path,
    target_data_root: Path, bootstrap_resamples: int = 1000,
) -> dict:
    if seed < 0 or bootstrap_resamples < 0:
        raise ValueError("Seed and bootstrap resamples must be non-negative")
    output = run_dir / "metrics.json"
    if output.exists():
        raise FileExistsError(f"Metrics already exist; refusing to overwrite: {output}")
    fold = FOLDS[fold_name]
    completion, digest = verify_completed_run(run_dir, fold_name, seed)
    manifest = load_score_manifest(run_dir, fold_name, seed, digest)
    source_dev = {}
    for domain in fold["sources"]:
        path = run_dir / f"source_dev_{domain}.csv"
        verify_score_digest(path, manifest)
        expected = canonical_score_labels(domain, "dev", source_data_root)
        source_dev[domain] = read_score_csv(path, domain, expected_labels=expected)
    calibration = calibrate_source_scores(source_dev)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    # Target membership and labels are opened only after source calibration succeeds.
    target_path = run_dir / "target_scores.csv"
    verify_score_digest(target_path, manifest)
    expected = canonical_score_labels(fold["target"], fold["target_split"], target_data_root)
    target_labels, target_scores = read_score_csv(target_path, fold["target"], expected_labels=expected)
    report = evaluate_transfer(
        source_dev,
        target_labels,
        target_scores,
        bootstrap_resamples=bootstrap_resamples,
    )
    report.update({
        "model": run_dir.parent.parent.name, "fold": fold_name, "seed": seed,
        "sources": list(fold["sources"]), "target_domain": fold["target"], "target_split": fold["target_split"],
        "checkpoint_sha256": digest, "checkpoint_epoch": completion["best_epoch"],
        "score_sha256": dict(manifest["score_sha256"]),
        "source_data_root": str(source_data_root), "target_data_root": str(target_data_root),
        "evaluator_sha256": checkpoint_digest(Path(__file__)),
    })
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen LODO run from common score CSVs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/salt/datasets/audio-deepfake"))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    report = evaluate_run(
        args.run_dir, args.fold, seed=args.seed, source_data_root=args.source_data_root,
        target_data_root=args.target_data_root, bootstrap_resamples=args.bootstrap_resamples,
    )
    print(f"Target EER: {report['target']['eer'] * 100:.4f}%")
    print(f"Target AUROC: {report['target']['auroc']:.6f}")
    print(f"Full report: {args.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
