"""Aggregate completed LODO metric files without pooling target datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_deepfake_detection.protocol import FOLDS


def flat_metrics(report: dict) -> dict[str, float]:
    values = {
        "target.eer": report["target"]["eer"],
        "target.auroc": report["target"]["auroc"],
    }
    for branch in ("strict_raw_transfer", "unlabeled_zscore_transfer"):
        for rate, operating in report[branch]["target_operating_points"].items():
            for metric in ("apcer", "bpcer", "acer"):
                values[f"{branch}.{rate}.{metric}"] = operating[metric]
    return values


def summarize(root: Path, model: str, seeds: tuple[int, ...]) -> dict:
    if len(seeds) < 2:
        raise ValueError("At least two seeds are needed for sample standard deviation")
    by_fold = {}
    for fold_name, fold in FOLDS.items():
        runs = []
        for seed in seeds:
            path = root / model / fold_name / str(seed) / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["fold"] != fold_name or report["target_domain"] != fold["target"]:
                raise ValueError(f"Wrong fold/target in {path}")
            if "seed" in report and report["seed"] != seed:
                raise ValueError(f"Wrong seed in {path}")
            if "confidence_intervals" not in report:
                raise ValueError(f"Final run lacks confidence intervals: {path}")
            runs.append(flat_metrics(report))
        metric_names = runs[0].keys()
        if any(run.keys() != metric_names for run in runs[1:]):
            raise ValueError(f"Metric columns differ across {fold_name} seeds")
        by_fold[fold_name] = {
            name: {
                "mean": float(np.mean([run[name] for run in runs])),
                "sample_std": float(np.std([run[name] for run in runs], ddof=1)),
            }
            for name in metric_names
        }
    equal_fold_mean = {
        name: float(np.mean([by_fold[fold][name]["mean"] for fold in FOLDS]))
        for name in by_fold["f1"]
    }
    return {
        "model": model,
        "seeds": list(seeds),
        "per_fold": by_fold,
        "equal_weight_four_fold_mean": equal_fold_mean,
        "target_datasets_pooled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize three-seed, four-fold LODO results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(1234, 2345, 3456))
    args = parser.parse_args()
    output = args.root / args.model / "summary.json"
    summary = summarize(args.root, args.model, tuple(args.seeds))
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
