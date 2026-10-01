"""Shared, spoof-oriented metrics for every detector in the LODO study.

Labels: 0 = bona fide, 1 = spoof. Larger scores mean more spoof-like.
Normalization is for score-transfer evaluation, never for training or checkpoint
selection. The normalized branch uses unlabeled target-score statistics and is
therefore transductive, not strict zero-shot.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


APCER_RATES = (0.01, 0.05, 0.10)
MIN_SCORE_STD = 1e-12


def _scores(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty, finite 1-D array")
    return scores


def _pair(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    scores_array = _scores(scores)
    if labels_array.ndim != 1 or labels_array.size != scores_array.size:
        raise ValueError("Labels and scores must be equal-length 1-D arrays")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("Labels must contain only 0 (bona fide) or 1 (spoof)")
    return labels_array.astype(np.int64), scores_array


def _both_classes(labels: np.ndarray) -> None:
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("Both bona fide and spoof examples are required")


def compute_eer(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array, scores_array = _pair(labels, scores)
    _both_classes(labels_array)
    fpr, tpr, _ = roc_curve(labels_array, scores_array, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossing = np.flatnonzero(difference >= 0)[0]
    if crossing == 0:
        return float(fpr[0])
    left = crossing - 1
    fraction = -difference[left] / (difference[crossing] - difference[left])
    return float(fpr[left] + fraction * (fpr[crossing] - fpr[left]))


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array, scores_array = _pair(labels, scores)
    _both_classes(labels_array)
    return float(roc_auc_score(labels_array, scores_array))


def compute_apcer(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    labels_array, scores_array = _pair(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    spoof = scores_array[labels_array == 1]
    if spoof.size == 0:
        raise ValueError("APCER requires spoof examples")
    return float(np.mean(spoof < threshold))


def compute_bpcer(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    labels_array, scores_array = _pair(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    bona_fide = scores_array[labels_array == 0]
    if bona_fide.size == 0:
        raise ValueError("BPCER requires bona fide examples")
    return float(np.mean(bona_fide >= threshold))


def compute_acer(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    return (compute_apcer(labels, scores, threshold) + compute_bpcer(labels, scores, threshold)) / 2.0


def calibrate_apcer_thresholds(
    labels: Sequence[int],
    scores: Sequence[float],
    rates: Sequence[float] = APCER_RATES,
) -> dict[str, float]:
    labels_array, scores_array = _pair(labels, scores)
    spoof = np.sort(scores_array[labels_array == 1])
    if spoof.size == 0:
        raise ValueError("Threshold calibration requires spoof examples")
    thresholds = {}
    for rate in rates:
        if not 0 <= rate < 1:
            raise ValueError("APCER rates must be in [0, 1)")
        # With spoof predicted for score >= threshold, this order statistic
        # gives source APCER no greater than the requested rate, including ties.
        thresholds[f"{100 * rate:g}%"] = float(spoof[int(np.floor(rate * spoof.size))])
    return thresholds


def zscore_stats(scores: Sequence[float]) -> dict[str, float]:
    values = _scores(scores)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= MIN_SCORE_STD:
        raise ValueError(f"Cannot z-score a constant or near-constant score distribution (std={std})")
    return {"mean": mean, "std": std}


def normalize_scores(scores: Sequence[float], stats: Mapping[str, float]) -> np.ndarray:
    values = _scores(scores)
    mean = float(stats["mean"])
    std = float(stats["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= MIN_SCORE_STD:
        raise ValueError("Invalid z-score mean or standard deviation")
    normalized = (values - mean) / std
    return _scores(normalized)


def _operating_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    spoof = scores[labels == 1]
    bona_fide = scores[labels == 0]
    if spoof.size == 0 or bona_fide.size == 0:
        raise ValueError("Operating-point metrics require both classes")
    result = {}
    for rate, threshold in thresholds.items():
        apcer = float(np.mean(spoof < threshold))
        bpcer = float(np.mean(bona_fide >= threshold))
        result[rate] = {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2.0}
    return result


def _bootstrap_intervals(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    normalized_scores: np.ndarray,
    raw_thresholds: Mapping[str, float],
    normalized_thresholds: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    bona_indices = np.flatnonzero(labels == 0)
    spoof_indices = np.flatnonzero(labels == 1)
    values: dict[str, list[float]] = {"eer": [], "auroc": []}
    for branch, thresholds in (("strict_raw", raw_thresholds), ("unlabeled_zscore", normalized_thresholds)):
        for rate in thresholds:
            for metric in ("apcer", "bpcer", "acer"):
                values[f"{branch}.{rate}.{metric}"] = []

    for _ in range(resamples):
        indices = np.concatenate((
            rng.choice(bona_indices, size=bona_indices.size, replace=True),
            rng.choice(spoof_indices, size=spoof_indices.size, replace=True),
        ))
        sample_labels = labels[indices]
        sample_raw = raw_scores[indices]
        sample_normalized = normalized_scores[indices]
        values["eer"].append(compute_eer(sample_labels, sample_raw))
        values["auroc"].append(compute_auroc(sample_labels, sample_raw))
        for branch, sample_scores, thresholds in (
            ("strict_raw", sample_raw, raw_thresholds),
            ("unlabeled_zscore", sample_normalized, normalized_thresholds),
        ):
            for rate, rates in _operating_metrics(sample_labels, sample_scores, thresholds).items():
                for metric, value in rates.items():
                    values[f"{branch}.{rate}.{metric}"].append(value)

    intervals = {
        key: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        for key, samples in values.items()
    }
    return {
        "method": "class-stratified percentile bootstrap",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "target_zscore_stats_fixed_within_resamples": True,
        "intervals": intervals,
    }


def evaluate_transfer(
    source_dev: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    target_labels: Sequence[int],
    target_scores: Sequence[float],
    *,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 2026,
) -> dict[str, object]:
    """Evaluate one frozen model/fold/seed; source-only calibration precedes target metrics."""
    if not source_dev:
        raise ValueError("At least one source-development domain is required")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")

    source_eers = {}
    source_stats = {}
    raw_labels, raw_scores, normalized_scores = [], [], []
    for domain, (labels, scores) in source_dev.items():
        domain_labels, domain_scores = _pair(labels, scores)
        _both_classes(domain_labels)
        source_eers[domain] = compute_eer(domain_labels, domain_scores)
        stats = zscore_stats(domain_scores)
        source_stats[domain] = stats
        raw_labels.append(domain_labels)
        raw_scores.append(domain_scores)
        normalized_scores.append(normalize_scores(domain_scores, stats))

    pooled_labels = np.concatenate(raw_labels)
    raw_thresholds = calibrate_apcer_thresholds(pooled_labels, np.concatenate(raw_scores))
    normalized_thresholds = calibrate_apcer_thresholds(pooled_labels, np.concatenate(normalized_scores))

    target_labels_array, target_raw = _pair(target_labels, target_scores)
    _both_classes(target_labels_array)
    target_stats = zscore_stats(target_raw)  # No target labels enter this calculation.
    target_normalized = normalize_scores(target_raw, target_stats)

    report: dict[str, object] = {
        "label_convention": "0=bona_fide,1=spoof; higher_score=spoof",
        "source_dev": {
            "eer_by_domain": source_eers,
            "macro_eer": float(np.mean(list(source_eers.values()))),
        },
        "target": {
            "n_bona_fide": int(np.sum(target_labels_array == 0)),
            "n_spoof": int(np.sum(target_labels_array == 1)),
            "eer": compute_eer(target_labels_array, target_raw),
            "auroc": compute_auroc(target_labels_array, target_raw),
        },
        "strict_raw_transfer": {
            "thresholds": raw_thresholds,
            "target_operating_points": _operating_metrics(target_labels_array, target_raw, raw_thresholds),
        },
        "unlabeled_zscore_transfer": {
            "transductive": True,
            "thresholds": normalized_thresholds,
            "target_operating_points": _operating_metrics(target_labels_array, target_normalized, normalized_thresholds),
        },
        "normalization": {
            "ddof": 0,
            "minimum_standard_deviation": MIN_SCORE_STD,
            "source_dev": source_stats,
            "target_unlabeled": target_stats,
        },
    }
    if bootstrap_resamples:
        report["confidence_intervals"] = _bootstrap_intervals(
            target_labels_array,
            target_raw,
            target_normalized,
            raw_thresholds,
            normalized_thresholds,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    return report


def write_score_csv(path: str | Path, rows: Iterable[tuple[str, str, int, float]]) -> None:
    """Write a complete score file atomically; refuse to replace a prior result."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Score file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("utterance_id", "dataset", "label", "raw_score"))
        for utterance_id, dataset, label, score in rows:
            if label not in (0, 1) or not np.isfinite(score):
                raise ValueError("Cannot write a non-binary label or non-finite score")
            writer.writerow((utterance_id, dataset, label, float(score)))
    os.replace(temporary, destination)


def read_score_csv(path: str | Path, expected_dataset: str) -> tuple[np.ndarray, np.ndarray]:
    labels, scores, seen_ids = [], [], set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"utterance_id", "dataset", "label", "raw_score"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required score columns in {path}")
        for row in reader:
            if row["dataset"] != expected_dataset:
                raise ValueError(f"Wrong dataset in {path}: {row['dataset']}")
            utterance_id = row["utterance_id"]
            if not utterance_id or utterance_id in seen_ids:
                raise ValueError(f"Blank or duplicate utterance ID in {path}: {utterance_id}")
            seen_ids.add(utterance_id)
            labels.append(int(row["label"]))
            scores.append(float(row["raw_score"]))
    return _pair(labels, scores)
