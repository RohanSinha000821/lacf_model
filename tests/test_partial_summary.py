import copy
import json

import pytest

from audio_deepfake_detection.metrics import evaluate_transfer
from audio_deepfake_detection.protocol import FOLDS
from scripts.summarize_results import summarize


def test_partial_then_complete_summary_preserves_first_three(tmp_path):
    template = evaluate_transfer({"source": ([0, 0, 1, 1], [-2, -1, 1, 2])},
                                 [0, 0, 1, 1], [-2, -1, 1, 2], bootstrap_resamples=3)
    def add(fold_name):
        fold = FOLDS[fold_name]
        report = copy.deepcopy(template)
        report.update(model="wavlm_bs96", fold=fold_name, seed=1234, sources=list(fold["sources"]),
                      target_domain=fold["target"], target_split=fold["target_split"])
        directory = tmp_path / "wavlm_bs96" / fold_name / "1234"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps(report))
    for fold in ("f1", "f2", "f3"):
        add(fold)
    before = {fold: (tmp_path / "wavlm_bs96" / fold / "1234/metrics.json").read_bytes()
              for fold in ("f1", "f2", "f3")}
    partial = summarize(tmp_path, "wavlm_bs96", non_final=True, folds=("f1", "f2", "f3"))
    assert partial["publication_summary"] is False
    assert partial["four_fold_complete"] is False and partial["missing_folds"] == ["f4"]
    assert partial["equal_weight_four_fold_mean"] is None
    assert partial["equal_weight_available_fold_mean"]["target.auroc"] == 1
    with pytest.raises(ValueError, match="all four folds"):
        summarize(tmp_path, "wavlm_bs96", folds=("f1", "f2", "f3"))
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path, "wavlm_bs96")
    add("f4")
    complete = summarize(tmp_path, "wavlm_bs96")
    assert complete["publication_summary"] is True and complete["four_fold_complete"] is True
    assert complete["missing_folds"] == []
    assert complete["per_fold"]["f1"] == partial["per_fold"]["f1"]
    assert all((tmp_path / "wavlm_bs96" / fold / "1234/metrics.json").read_bytes() == saved
               for fold, saved in before.items())


@pytest.mark.parametrize("folds", [(), ("f1", "f1"), ("f5",)])
def test_partial_summary_rejects_invalid_fold_selection(tmp_path, folds):
    with pytest.raises(ValueError, match="distinct valid folds"):
        summarize(tmp_path, "wavlm_bs96", non_final=True, folds=folds)
