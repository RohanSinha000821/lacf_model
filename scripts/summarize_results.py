"""Aggregate completed LODO metric files without pooling target datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_deepfake_detection.protocol import FOLDS, final_seeds_for


def flat_metrics(report: dict) -> dict[str, float]:
    values = {
        "target.eer": report["target"]["eer"],
        "target.auroc": report["target"]["auroc"],
    }
    for branch in ("strict_raw_transfer", "unlabeled_zscore_transfer"):
        if (set(report[branch]["target_operating_points"]) != {"1%", "5%", "10%"}
                or set(report[branch]["thresholds"]) != {"1%", "5%", "10%"}):
            raise ValueError(f"Unexpected operating rates in {branch}")
        for rate, operating in report[branch]["target_operating_points"].items():
            if set(operating) != {"apcer", "bpcer", "acer"}:
                raise ValueError(f"Unexpected operating metrics in {branch}/{rate}")
            for metric in ("apcer", "bpcer", "acer"):
                values[f"{branch}.{rate}.{metric}"] = operating[metric]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
           for value in values.values()):
        raise ValueError("Metrics must be finite fractions in [0, 1]")
    return values


def summarize(root: Path, model: str, seeds: tuple[int, ...] | None = None, *, non_final=False, folds=None) -> dict:
    required_seeds = final_seeds_for(model)
    if seeds is None:
        seeds = required_seeds
    if not non_final and seeds != required_seeds:
        raise ValueError(f"Publication summary for {model} requires exactly seeds {required_seeds}; use non-final mode for exploration")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("At least one distinct non-negative seed is required")
    selected_folds = tuple(FOLDS) if folds is None else tuple(folds)
    if not selected_folds or len(set(selected_folds)) != len(selected_folds) or any(f not in FOLDS for f in selected_folds):
        raise ValueError("Select distinct valid folds")
    if not non_final and set(selected_folds) != set(FOLDS):
        raise ValueError("A publication summary requires all four folds; use non-final mode for partial results")
    selected_folds = tuple(f for f in FOLDS if f in selected_folds)
    by_fold = {}
    expected_metric_names = None
    for fold_name in selected_folds:
        fold = FOLDS[fold_name]
        runs = []
        for seed in seeds:
            path = root / model / fold_name / str(seed) / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("fold") != fold_name or report.get("target_domain") != fold["target"]
                    or report.get("target_split") != fold["target_split"]
                    or report.get("sources") != list(fold["sources"])):
                raise ValueError(f"Wrong fold/target in {path}")
            if type(report.get("seed")) is not int or report["seed"] != seed:
                raise ValueError(f"Missing or wrong seed in {path}")
            if report.get("model") != model:
                raise ValueError(f"Missing or wrong model in {path}")
            ci = report.get("confidence_intervals", {})
            if not ci.get("intervals") or ci.get("resamples", 0) <= 0:
                raise ValueError(f"Final run lacks confidence intervals: {path}")
            values = flat_metrics(report)
            ci_names = {name.removeprefix("target.").replace("strict_raw_transfer.", "strict_raw.")
                        .replace("unlabeled_zscore_transfer.", "unlabeled_zscore.") for name in values}
            if set(ci["intervals"]) != ci_names:
                raise ValueError(f"Confidence interval metric structure differs in {path}")
            for interval in ci["intervals"].values():
                if (not isinstance(interval, list) or len(interval) != 2
                        or any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in interval)
                        or not 0 <= interval[0] <= interval[1] <= 1):
                    raise ValueError(f"Invalid confidence interval in {path}")
            if (ci.get("resampling_unit") != "target_observation" or ci.get("class_stratified") is not True
                    or ci.get("source_thresholds_fixed_within_resamples") is not True
                    or ci.get("source_normalization_stats_fixed_within_resamples") is not True
                    or ci.get("target_zscore_stats_fixed_within_resamples") is not True
                    or ci.get("method") != "class-stratified percentile bootstrap"
                    or ci.get("percentiles") != [2.5, 97.5]
                    or ci.get("confidence_level") != 0.95 or ci.get("seed") != 2026):
                raise ValueError(f"Confidence interval assumptions differ in {path}")
            if expected_metric_names is None:
                expected_metric_names = values.keys()
            if values.keys() != expected_metric_names:
                raise ValueError(f"Metric structures differ in {path}")
            runs.append(values)
        metric_names = runs[0].keys()
        if any(run.keys() != metric_names for run in runs[1:]):
            raise ValueError(f"Metric columns differ across {fold_name} seeds")
        by_fold[fold_name] = {
            name: {
                "mean": float(np.mean([run[name] for run in runs])),
                "sample_std": float(np.std([run[name] for run in runs], ddof=1)) if len(runs) > 1 else None,
            }
            for name in metric_names
        }
    equal_fold_mean = {
        name: float(np.mean([by_fold[fold][name]["mean"] for fold in selected_folds]))
        for name in by_fold[selected_folds[0]]
    }
    return {
        "model": model,
        "seeds": list(seeds),
        "training_seed_count": len(seeds),
        "across_seed_variability_estimated": len(seeds) > 1,
        "seed_policy": "single_seed_compute_budget" if model == "wavlm_bs96" and seeds == (1234,) else "multi_seed" if len(seeds) > 1 else "single_seed_exploratory",
        "per_fold": by_fold,
        "completed_folds": list(selected_folds),
        "missing_folds": [f for f in FOLDS if f not in selected_folds],
        "four_fold_complete": len(selected_folds) == len(FOLDS),
        "equal_weight_four_fold_mean": equal_fold_mean if len(selected_folds) == len(FOLDS) else None,
        "equal_weight_available_fold_mean": equal_fold_mean,
        "target_datasets_pooled": False,
        "publication_summary": not non_final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize four-fold LODO results using the model's declared seed policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", help="Defaults to seed 1234 for wavlm_bs96; three protocol seeds for other models")
    parser.add_argument("--non-final", action="store_true", help="Allow exploratory seed lists; write exploratory_summary.json")
    parser.add_argument("--folds", choices=tuple(FOLDS), nargs="+", help="Explicit subset requires --non-final; final summary still requires four folds")
    args = parser.parse_args()
    if args.folds and not args.non_final and set(args.folds) != set(FOLDS):
        parser.error("Partial fold selection requires --non-final")
    filename = ("partial_summary_" + "_".join(f for f in FOLDS if f in args.folds) + ".json"
                if args.non_final and args.folds and len(set(args.folds)) < len(FOLDS)
                else "exploratory_summary.json" if args.non_final else "summary.json")
    output = args.root / args.model / filename
    if output.exists():
        raise FileExistsError(f"Summary already exists; refusing to overwrite: {output}")
    summary = summarize(args.root, args.model, tuple(args.seeds) if args.seeds is not None else None, non_final=args.non_final, folds=args.folds)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
