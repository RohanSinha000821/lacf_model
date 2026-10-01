import numpy as np
import pytest

from audio_deepfake_detection.metrics import (
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
    assert set(FOLDS) == {"f1", "f2", "f3", "f4"}
    assert {fold["target"] for fold in FOLDS.values()} == {"asv2019", "asv5", "cfad", "speechfake"}
    for fold in FOLDS.values():
        assert len(fold["sources"]) == 3
        assert fold["target"] not in fold["sources"]
