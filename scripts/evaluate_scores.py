"""Run the common LODO metrics on score CSVs from any detector.

Expected files in --run-dir: source_dev_<dataset>.csv for each source domain,
and target_scores.csv. Each file has utterance_id,dataset,label,raw_score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_deepfake_detection.metrics import evaluate_transfer, read_score_csv
from audio_deepfake_detection.protocol import FOLDS


def evaluate_run(run_dir: Path, fold_name: str, *, bootstrap_resamples: int = 1000) -> dict:
    fold = FOLDS[fold_name]
    source_dev = {
        domain: read_score_csv(run_dir / f"source_dev_{domain}.csv", domain)
        for domain in fold["sources"]
    }
    target_labels, target_scores = read_score_csv(run_dir / "target_scores.csv", fold["target"])
    report = evaluate_transfer(
        source_dev,
        target_labels,
        target_scores,
        bootstrap_resamples=bootstrap_resamples,
    )
    report["fold"] = fold_name
    report["sources"] = list(fold["sources"])
    report["target_domain"] = fold["target"]
    report["target_split"] = fold["target_split"]
    output = run_dir / "metrics.json"
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen LODO run from common score CSVs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    report = evaluate_run(args.run_dir, args.fold, bootstrap_resamples=args.bootstrap_resamples)
    print(f"Target EER: {report['target']['eer'] * 100:.4f}%")
    print(f"Target AUROC: {report['target']['auroc']:.6f}")
    print(f"Full report: {args.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
