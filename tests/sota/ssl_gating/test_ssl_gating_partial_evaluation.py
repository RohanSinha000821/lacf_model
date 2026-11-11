"""Synthetic CPU checks for partial SSL identity, source gates and native scores."""

import copy
import json
import sys

import numpy as np
import pytest
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, dataset_root
from audio_deepfake_detection.sota.ssl_gating import SSLFeatureGating
from scripts.ssl_gating import evaluate_partial as evaluate


@pytest.fixture(params=("f3", "f4"))
def source_run(tmp_path, monkeypatch, request):
    fold = request.param
    run = tmp_path / "outputs" / evaluate.RUN_FAMILY / fold / "1234"
    run.mkdir(parents=True)
    config = {"fold": fold, "seed": 1234, "run_name": evaluate.RUN_FAMILY,
              "model_name": "PartialSSLFeatureGating", "model_config": evaluate.PARTIAL_MODEL_CONFIG,
              "precision": "BF16_autocast_FP32_pooling_losses_scores", "train_max_seconds": 20,
              "eval_batch_size": 1, "selection": "strict_equal_domain_macro_source_dev_eer_ties_earlier",
              "trainable_parameters": 53029762, "total_parameters": 318083586,
              "sources": list(FOLDS[fold]["sources"]), "target_domain": FOLDS[fold]["target"], "target_access": False,
              "dataset_counts": {d: {"dev": 4} for d in FOLDS[fold]["sources"]}}
    records, scores = {}, {}
    for domain in config["sources"]:
        records[domain] = [(str(dataset_root(tmp_path, domain) / f"{i}.wav"), i % 2) for i in range(4)]
        labels, values = np.array([0, 1, 0, 1]), np.array([-2., 2., -1., 1.])
        order = [2, 0, 3, 1]
        scores[domain] = (labels[order], values[order])
        write_score_csv(run/f"source_dev_{domain}.csv",
                        ((f"{i}.wav", domain, i % 2, float(values[i])) for i in order))
    calibration = calibrate_source_scores(scores)
    (run/"source_calibration.json").write_text(json.dumps(calibration))
    completion = {"fold": fold, "seed": 1234, "best_epoch": 9, "final_epoch": 10,
                  "selected_checkpoint_reloaded_and_source_verified": True, "target_access": False,
                  "best_macro_source_dev_eer": 0., "source_calibration_sha256": checkpoint_digest(run/"source_calibration.json"),
                  "source_scores_sha256": {p.name: checkpoint_digest(p) for p in run.glob("source_dev_*.csv")}}
    opened = []
    def reader(domain, split, root):
        opened.append((domain, split))
        assert split == "dev", "A failed source gate must never open the target"
        return records[domain]
    monkeypatch.setattr(evaluate, "read_dataset", reader)
    return run, config, completion, opened


@pytest.mark.parametrize("key,value", [("precision", "float32"), ("run_name", "ssl_gating_20s"),
                                      ("seed", 2345), ("target_domain", "cfad"),
                                      ("train_max_seconds", 10), ("trainable_parameters", 318083586)])
def test_reject_wrong_partial_recipe(source_run, key, value):
    _, config, completion, _ = source_run
    checkpoint = {"config": copy.deepcopy(config), "fold": config["fold"], "epoch": 9, "macro_dev_eer": 0.}
    evaluate.verify_checkpoint(checkpoint, completion, config["fold"], 1234)
    checkpoint["config"][key] = value
    with pytest.raises(ValueError, match="identity, recipe"):
        evaluate.verify_checkpoint(checkpoint, completion, config["fold"], 1234)


def test_source_scores_reproduce_frozen_calibration_without_overwriting(source_run):
    run, config, completion, opened = source_run
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    sources, records = evaluate.verified_sources(run, completion, config, run.parents[3])
    assert len(sources) == len(records) == 3
    assert all(values.tolist() == [-1., -2., 1., 2.] for _, values in sources.values())
    assert opened == [(d, "dev") for d in config["sources"]]
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


def test_config_json_key_conversion_preserves_real_differences_and_source_guards():
    config = {"frontend_config": {"id2label": {0: "LABEL_0", 1: "LABEL_1"}},
              "precision": "BF16", "source_sha256": {"model.py": "original"}}
    stored = json.loads(json.dumps(config))
    evaluate.verify_run_config(config, stored, {"model.py": "original"})
    with pytest.raises(ValueError, match="configuration or guarded runtime"):
        evaluate.verify_run_config(config, stored, {"model.py": "changed"})
    stored["precision"] = "FP32"
    with pytest.raises(ValueError, match="configuration or guarded runtime"):
        evaluate.verify_run_config(config, stored, {"model.py": "original"})


def test_source_digest_and_calibration_mismatch_block_target(source_run):
    run, config, completion, opened = source_run
    wrong = copy.deepcopy(completion)
    wrong["best_macro_source_dev_eer"] = .2
    with pytest.raises(ValueError, match="Source-dev macro"):
        evaluate.verified_sources(run, wrong, config, run.parents[3])
    assert all(split == "dev" for _, split in opened)
    path = run/f"source_dev_{config['sources'][0]}.csv"
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="Recorded source scores changed"):
        evaluate.verified_sources(run, completion, config, run.parents[3])


def test_main_source_failure_never_constructs_model_or_opens_target(source_run, monkeypatch):
    run, config, completion, opened = source_run
    config["source_sha256"] = {}
    (run/"config.json").write_text(json.dumps(config))
    checkpoint = {"config": config, "fold": config["fold"], "epoch": 9, "macro_dev_eer": 0.}
    torch.save(checkpoint, run/"best.pt")
    completion["best_checkpoint_sha256"] = checkpoint_digest(run/"best.pt")
    (run/"training_complete.json").write_text(json.dumps(completion))
    (run/"source_dev_cfad.csv").write_text("changed")
    monkeypatch.setenv("PROJECT_ROOT", str(run.parents[3]))
    monkeypatch.setattr(evaluate, "source_guards", lambda: {})
    monkeypatch.setattr(sys, "argv", ["evaluate_partial", "--fold", config["fold"], "--source-data-root", str(run.parents[3])])
    monkeypatch.setattr(evaluate.PartialSSLFeatureGating, "from_frontend_config",
                        lambda *_: pytest.fail("Source failure must precede model construction"))
    with pytest.raises(ValueError, match="Recorded source scores changed"):
        evaluate.main()
    assert all(split == "dev" for _, split in opened)
    assert not (run/"target_scores.csv").exists()


def test_scoring_uses_bf16_fp32_score_subtraction_and_keeps_full_long_input(tmp_path):
    class Model(torch.nn.Module):
        spoof_score = staticmethod(SSLFeatureGating.spoof_score)
        def forward(self, waves):
            assert waves.shape == (1, 320017)  # beyond the training-only 20 s limit
            assert torch.is_autocast_enabled("cpu")
            return torch.tensor([[1., 3.]], dtype=torch.bfloat16)
    path = dataset_root(tmp_path, "asv5") / "long.wav"
    rows = list(evaluate.score_rows(Model(), [(torch.ones(1, 320017), torch.tensor([1]))],
                                    [(str(path), 1)], "asv5", tmp_path, torch.device("cpu")))
    assert rows == [("long.wav", "asv5", 1, 2.)]
    with pytest.raises(RuntimeError, match="Incomplete target"):
        list(evaluate.score_rows(Model(), [], [(str(path), 1)], "asv5", tmp_path, torch.device("cpu")))


@pytest.mark.parametrize("mismatch", [False, True])
def test_f3_f4_fresh_bf16_source_parity_preserves_exports(source_run, monkeypatch, mismatch):
    run, config, completion, opened = source_run
    data_root = run.parents[3]
    _, records = evaluate.verified_sources(run, completion, config, data_root)
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[0.], [1.]]))
        def forward(self, waves):
            return self.linear(waves)
        spoof_score = staticmethod(SSLFeatureGating.spoof_score)
    # Same exact BF16/FP32 score subtraction as training, including F4 coverage
    # from the independently developed SSH2 exporter tests.
    values = [-2., 2., -1., 1.]
    monkeypatch.setattr(evaluate, "FullAudioDataset", lambda rows: [
        (torch.tensor([99. if mismatch else values[int(__import__('pathlib').Path(path).stem)]]), label)
        for path, label in rows])
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    if mismatch:
        with pytest.raises(ValueError, match="Fresh BF16 source score differs"):
            evaluate.verify_source_parity(Detector(), records, run, data_root, torch.device("cpu"))
    else:
        evidence = evaluate.verify_source_parity(Detector(), records, run, data_root, torch.device("cpu"))
        assert len(evidence) == 6 and all(row["exact_match"] for row in evidence)
    assert all(split == "dev" for _, split in opened)
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
