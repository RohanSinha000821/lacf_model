"""Common frozen-calibration P2–P4 reports, independent of detector architecture."""

import numpy as np

from .metrics import (
    _pair, _bootstrap_intervals, compute_eer, compute_auroc, compute_apcer,
    compute_bpcer, normalize_scores, zscore_stats,
)


def evaluate_frozen(calibration, labels, scores, *, target_stats=None,
                    bootstrap_resamples=1000, bootstrap_seed=2026):
    """Never recalibrate on a target subgroup; missing-class metrics are null."""
    if bootstrap_resamples < 0:
        raise ValueError("Bootstrap resamples must be nonnegative")
    labels, raw = _pair(labels, scores)
    stats = zscore_stats(raw) if target_stats is None else target_stats
    normalized = normalize_scores(raw, stats)
    bona, spoof = bool(np.any(labels == 0)), bool(np.any(labels == 1))
    report = {
        "target": {"n_bona_fide": int(np.sum(labels == 0)), "n_spoof": int(np.sum(labels == 1)),
                   "eer": compute_eer(labels, raw) if bona and spoof else None,
                   "auroc": compute_auroc(labels, raw) if bona and spoof else None},
        "source_dev": calibration["source_dev"],
        "normalization": {"source_dev": calibration["source_stats"], "target_unlabeled": stats,
                          "ddof": 0, "statistics_fixed_for_subgroups_and_bootstrap": True},
        "undefined_metrics_reason": None if bona and spoof else "Only one class is present; no ranking or ACER claim",
    }
    for branch, values, key in (("strict_raw_transfer", raw, "raw_thresholds"),
                               ("unlabeled_zscore_transfer", normalized, "normalized_thresholds")):
        points = {}
        for rate, threshold in calibration[key].items():
            apcer = compute_apcer(labels, values, threshold) if spoof else None
            bpcer = compute_bpcer(labels, values, threshold) if bona else None
            points[rate] = {"apcer": apcer, "bpcer": bpcer,
                            "acer": (apcer + bpcer) / 2 if bona and spoof else None}
        report[branch] = {"thresholds": dict(calibration[key]), "target_operating_points": points}
    report["unlabeled_zscore_transfer"]["transductive"] = True
    if bootstrap_resamples and bona and spoof:
        report["confidence_intervals"] = _bootstrap_intervals(
            labels, raw, normalized, calibration["raw_thresholds"], calibration["normalized_thresholds"],
            resamples=bootstrap_resamples, seed=bootstrap_seed)
    elif bootstrap_resamples:
        rng = np.random.default_rng(bootstrap_seed)
        metric = "apcer" if spoof else "bpcer"
        function = compute_apcer if spoof else compute_bpcer
        intervals = {}
        # Ranking is undefined, but class-conditional error still has a valid interval.
        estimates = {}
        for _ in range(bootstrap_resamples):
            index = rng.integers(0, len(raw), len(raw))
            for branch, values, key in (("strict_raw", raw, "raw_thresholds"),
                                       ("unlabeled_zscore", normalized, "normalized_thresholds")):
                for rate, threshold in calibration[key].items():
                    estimates.setdefault(f"{branch}.{rate}.{metric}", []).append(
                        function(labels[index], values[index], threshold))
        intervals = {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in estimates.items()}
        report["confidence_intervals"] = {
            "method": "single-class percentile bootstrap", "resamples": bootstrap_resamples,
            "seed": bootstrap_seed, "confidence_level": 0.95, "intervals": intervals,
            "source_thresholds_fixed_within_resamples": True,
            "target_zscore_stats_fixed_within_resamples": True,
            "includes_training_or_calibration_uncertainty": False,
        }
    return report


def group_reports(examples, scores, calibration, dimensions, *, include_bona_reference=False,
                  target_stats=None, bootstrap_resamples=1000):
    """Generator/attack rows share all parent bona fide; codec rows keep own membership."""
    labels = np.asarray([item.label for item in examples])
    scores = np.asarray(scores, dtype=float)
    _pair(labels, scores)
    stats = zscore_stats(scores) if target_stats is None else target_stats
    result = {}
    for dimension in dimensions:
        values = sorted({item.groups[dimension] for item in examples
                         if dimension in item.groups and (not include_bona_reference or item.label == 1)})
        result[dimension] = {}
        for value in values:
            mask = np.asarray([item.groups.get(dimension) == value
                               or (include_bona_reference and item.label == 0) for item in examples])
            report = evaluate_frozen(calibration, labels[mask], scores[mask], target_stats=stats,
                                     bootstrap_resamples=bootstrap_resamples)
            report["membership"] = {"dimension": dimension, "value": value,
                                     "bona_fide_reference": "all_parent_bona_fide" if include_bona_reference else "same_group_only"}
            result[dimension][value] = report
    return result


def paired_raw_deltas(examples, scores, calibration, *, dimension, reference):
    """Match only declared original IDs; compare mean error per original (not per variant)."""
    buckets = {}
    for item, score in zip(examples, scores):
        condition = item.groups.get(dimension)
        if not item.pair_id or condition is None:
            continue
        key = item.pair_id
        bucket = buckets.setdefault(key, {"label": item.label, "conditions": {}})
        if bucket["label"] != item.label:
            raise ValueError(f"Paired robustness label conflict: {key}")
        bucket["conditions"].setdefault(condition, []).append(float(score))
    conditions = sorted({c for b in buckets.values() for c in b["conditions"]} - {reference})
    report = {}
    for condition in conditions:
        matched = [b for b in buckets.values() if reference in b["conditions"] and condition in b["conditions"]]
        points = {}
        for rate, threshold in calibration["raw_thresholds"].items():
            deltas = {0: [], 1: []}
            for bucket in matched:
                label = bucket["label"]
                def error(values):
                    return float(np.mean(np.asarray(values) < threshold if label else np.asarray(values) >= threshold))
                deltas[label].append(error(bucket["conditions"][condition]) - error(bucket["conditions"][reference]))
            points[rate] = {"delta_apcer": float(np.mean(deltas[1])) if deltas[1] else None,
                            "delta_bpcer": float(np.mean(deltas[0])) if deltas[0] else None}
        report[condition] = {"reference": reference, "n_matched_originals": len(matched),
                             "n_matched_bona_fide": sum(b["label"] == 0 for b in matched),
                             "n_matched_spoof": sum(b["label"] == 1 for b in matched),
                             "strict_raw_operating_point_deltas": points,
                             "variant_aggregation": "mean error per original before averaging originals",
                             "uncertainty": "paired deltas are descriptive; no independence-based CI attached"}
    return report
