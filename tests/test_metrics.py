import numpy as np
import pytest

from audio_deepfake_detection.metrics import (
    calibrate_source_scores,
    calibrate_apcer_thresholds,
    compute_acer,
    compute_apcer,
    compute_auroc,
    compute_bpcer,
    compute_eer,
    evaluate_transfer,
    normalize_scores,
    read_score_csv,
    write_score_csv,
    zscore_stats,
)
from audio_deepfake_detection.protocol import FOLDS


def test_ranking_metrics_and_ties():
    labels = [0, 0, 1, 1]
    assert compute_eer(labels, [0, 1, 2, 3]) == pytest.approx(0.0)
    assert compute_auroc(labels, [0, 1, 2, 3]) == pytest.approx(1.0)
    assert compute_eer(labels, [3, 2, 1, 0]) == pytest.approx(1.0)
    assert compute_auroc(labels, [3, 2, 1, 0]) == pytest.approx(0.0)
    assert compute_eer(labels, [1, 1, 1, 1]) == pytest.approx(0.5)


def test_operating_point_tie_rule_and_calibration():
    labels = [0, 0, 1, 1]
    scores = [2, 3, 1, 2]
    assert compute_apcer(labels, scores, 2) == pytest.approx(0.5)
    assert compute_bpcer(labels, scores, 2) == pytest.approx(1.0)
    assert compute_acer(labels, scores, 2) == pytest.approx(0.75)
    spoof_scores = list(range(10))
    thresholds = calibrate_apcer_thresholds([1] * 10, spoof_scores)
    assert thresholds == {"1%": 0.0, "5%": 0.0, "10%": 1.0}
    assert compute_apcer([1] * 10, spoof_scores, thresholds["10%"]) <= 0.10


def test_domainwise_normalization_and_rank_invariance():
    source = {
        "a": ([0, 0, 1, 1], [0, 1, 2, 3]),
        "b": ([0, 0, 1, 1], [100, 101, 102, 103]),
        "c": ([0, 0, 1, 1], [-100, -99, -98, -97]),
    }
    labels = [0, 0, 1, 1]
    scores = [200, 201, 202, 203]
    report = evaluate_transfer(source, labels, scores, bootstrap_resamples=20)
    assert report["target"]["eer"] == pytest.approx(0.0)
    assert report["target"]["auroc"] == pytest.approx(1.0)
    assert report["normalization"]["source_dev"]["a"]["mean"] == pytest.approx(1.5)
    assert report["normalization"]["source_dev"]["b"]["mean"] == pytest.approx(101.5)
    assert report["normalization"]["target_unlabeled"]["mean"] == pytest.approx(201.5)
    assert compute_eer(labels, normalize_scores(scores, zscore_stats(scores))) == pytest.approx(report["target"]["eer"])
    assert report["unlabeled_zscore_transfer"]["transductive"] is True
    assert report["confidence_intervals"]["resamples"] == 20
    assert "unlabeled_zscore.1%.acer" in report["confidence_intervals"]["intervals"]


def test_invalid_scores_and_constant_normalization_fail():
    with pytest.raises(ValueError, match="near-constant"):
        zscore_stats([2, 2, 2])
    with pytest.raises(ValueError, match="finite"):
        compute_eer([0, 1], [0, np.inf])
    with pytest.raises(ValueError, match="Both"):
        compute_auroc([1, 1], [0, 1])


def test_score_file_round_trip_and_no_overwrite(tmp_path):
    output = tmp_path / "scores.csv"
    write_score_csv(output, [("a.wav", "a", 0, -2.0), ("b.wav", "a", 1, 3.0)])
    labels, scores = read_score_csv(output, "a")
    assert labels.tolist() == [0, 1]
    assert scores.tolist() == [-2.0, 3.0]
    with pytest.raises(FileExistsError):
        write_score_csv(output, [])
    with pytest.raises(ValueError, match="Wrong dataset"):
        read_score_csv(output, "b")


def test_lodo_membership():
    assert FOLDS == {
        "f1": {"sources": ("asv2019", "asv5", "cfad"), "target": "speechfake", "target_split": "test"},
        "f2": {"sources": ("asv2019", "asv5", "speechfake"), "target": "cfad", "target_split": "test_unseen"},
        "f3": {"sources": ("asv2019", "cfad", "speechfake"), "target": "asv5", "target_split": "eval"},
        "f4": {"sources": ("asv5", "cfad", "speechfake"), "target": "asv2019", "target_split": "eval"},
    }
    assert "asv5" not in FOLDS["f3"]["sources"]
    assert set(FOLDS) == {"f1", "f2", "f3", "f4"}
    assert {fold["target"] for fold in FOLDS.values()} == {"asv2019", "asv5", "cfad", "speechfake"}
    for fold in FOLDS.values():
        assert len(fold["sources"]) == 3
        assert fold["target"] not in fold["sources"]


@pytest.mark.parametrize("metric", [compute_eer, compute_auroc])
@pytest.mark.parametrize("scores", [[np.nan, 1], [0, -np.inf], [0, np.inf]])
def test_nonfinite_ranking_scores_rejected(metric, scores):
    with pytest.raises(ValueError, match="finite"):
        metric([0, 1], scores)


@pytest.mark.parametrize("metric", [compute_eer, compute_auroc])
def test_one_class_ranking_rejected(metric):
    with pytest.raises(ValueError, match="Both"):
        metric([0, 0], [0, 1])


def test_threshold_boundary_and_tied_order_statistics():
    assert compute_apcer([1], [2], 2) == 0
    assert compute_bpcer([0], [2], 2) == 1
    labels = [1] * 100
    scores = [0] * 6 + [1] * 10 + list(range(2, 86))
    for rate, threshold in zip((0.01, 0.05, 0.10), calibrate_apcer_thresholds(labels, scores).values()):
        assert compute_apcer(labels, scores, threshold) <= rate


@pytest.mark.parametrize("rate", [-0.1, 1, np.nan, np.inf])
def test_invalid_calibration_rate(rate):
    with pytest.raises(ValueError):
        calibrate_apcer_thresholds([1], [2], [rate])


def test_domain_calibration_and_target_label_independence():
    source = {"a": ([0, 0, 1, 1], [0, 1, 2, 3]), "b": ([0, 0, 1, 1], [100, 102, 104, 106])}
    calibration = calibrate_source_scores(source)
    expected_z = np.concatenate([normalize_scores(scores, zscore_stats(scores)) for _, scores in source.values()])
    assert calibration["normalized_thresholds"] == calibrate_apcer_thresholds([0, 0, 1, 1] * 2, expected_z)
    target_scores = [200, 210, 230, 250]
    first = evaluate_transfer(source, [0, 0, 1, 1], target_scores, bootstrap_resamples=0)
    second = evaluate_transfer(source, [1, 1, 0, 0], target_scores, bootstrap_resamples=0)
    assert first["normalization"] == second["normalization"]
    assert first["normalization"]["ddof"] == 0
    assert first["normalization"]["target_unlabeled"]["std"] == np.std(target_scores, ddof=0)
    for branch in ("strict_raw_transfer", "unlabeled_zscore_transfer"):
        assert first[branch]["thresholds"] == second[branch]["thresholds"]
    normalized = normalize_scores(target_scores, zscore_stats(target_scores))
    assert compute_auroc([0, 0, 1, 1], normalized) == first["target"]["auroc"]


def test_near_constant_and_invalid_normalization():
    with pytest.raises(ValueError, match="near-constant"):
        zscore_stats([1, 1 + 1e-13])
    with pytest.raises(ValueError, match="Invalid"):
        normalize_scores([1, 2], {"mean": 1, "std": 0})


def test_bootstrap_fixed_calibration_and_class_counts(monkeypatch):
    import audio_deepfake_detection.metrics as metrics

    source = {"a": ([0, 0, 1, 1], [-1, 0, 1, 2])}
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([8, 10, 12, 11, 13])
    calibration = calibrate_source_scores(source)
    original_eer, original_operating = metrics.compute_eer, metrics._operating_metrics
    original_stats, original_calibrate = metrics.zscore_stats, metrics.calibrate_apcer_thresholds
    stats_calls, threshold_calls, target_calls = [], [], []

    def eer(sample_labels, sample_scores):
        if len(sample_labels) == 5:
            assert np.count_nonzero(np.asarray(sample_labels) == 0) == 3
            assert np.count_nonzero(np.asarray(sample_labels) == 1) == 2
            target_calls.append(np.asarray(sample_scores).copy())
        return original_eer(sample_labels, sample_scores)

    def operating(sample_labels, sample_scores, thresholds):
        assert thresholds in (calibration["raw_thresholds"], calibration["normalized_thresholds"])
        return original_operating(sample_labels, sample_scores, thresholds)

    def stats(values):
        stats_calls.append(np.asarray(values).copy())
        return original_stats(values)

    def thresholds(sample_labels, values):
        threshold_calls.append(np.asarray(values).copy())
        return original_calibrate(sample_labels, values)

    monkeypatch.setattr(metrics, "compute_eer", eer)
    monkeypatch.setattr(metrics, "_operating_metrics", operating)
    monkeypatch.setattr(metrics, "zscore_stats", stats)
    monkeypatch.setattr(metrics, "calibrate_apcer_thresholds", thresholds)
    report = evaluate_transfer(source, labels, scores, bootstrap_resamples=12)
    assert len(stats_calls) == 2  # One source and full target, never bootstrap statistics.
    assert len(threshold_calls) == 2  # One source calibration per branch.
    assert len(target_calls) == 13
    repeated = evaluate_transfer(source, labels, scores, bootstrap_resamples=12)
    assert report["confidence_intervals"] == repeated["confidence_intervals"]
    ci = report["confidence_intervals"]
    assert ci["source_thresholds_fixed_within_resamples"] is True
    assert ci["target_zscore_stats_fixed_within_resamples"] is True
    assert ci["resampling_unit"] == "target_observation"
    assert ci["percentiles"] == [2.5, 97.5]
    assert ci["intervals"]["eer"] == pytest.approx(np.percentile([
        original_eer(labels, sample) for sample in target_calls[1:13]
    ], [2.5, 97.5]))


@pytest.mark.parametrize("case,match", [
    ("wrong_ids", "Unexpected"), ("wrong_label", "Wrong label"), ("missing", "Missing"),
    ("extra", "Unexpected"), ("duplicate", "duplicate"), ("dataset", "Wrong dataset"),
])
def test_exact_score_membership_rejects_invalid_rows(tmp_path, case, match):
    rows = [("a.wav", "a", 0, -2), ("b.wav", "a", 1, 3)]
    if case == "wrong_ids":
        rows[0] = ("unexpected.wav", "a", 0, -2)
    elif case == "wrong_label":
        rows[0] = ("a.wav", "a", 1, -2)
    elif case == "missing":
        rows.pop()
    elif case == "extra":
        rows.append(("extra.wav", "a", 1, 4))
    elif case == "duplicate":
        rows[1] = rows[0]
    elif case == "dataset":
        rows[0] = ("a.wav", "b", 0, -2)
    path = tmp_path / "scores.csv"
    write_score_csv(path, rows)
    with pytest.raises(ValueError, match=match):
        read_score_csv(path, "a", expected_labels={"a.wav": 0, "b.wav": 1})


def test_exact_membership_is_order_independent(tmp_path):
    path = tmp_path / "scores.csv"
    write_score_csv(path, [("b.wav", "a", 1, 3), ("a.wav", "a", 0, -2)])
    labels, scores = read_score_csv(path, "a", expected_labels={"b.wav": 1, "a.wav": 0})
    assert labels.tolist() == [0, 1]
    assert scores.tolist() == [-2, 3]
