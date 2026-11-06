"""CPU-only scoring, artifact binding and shared-metric tests for new TF evaluation."""

import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.probes import FeatureBatches, score_probe
from audio_deepfake_detection.metrics import write_score_csv
from audio_deepfake_detection.protocol import checkpoint_digest, load_score_manifest, save_score_manifest
from audio_deepfake_detection.token_fusion.model import CONDITIONS, TokenFusion
from scripts.evaluate_scores import evaluate_run
from scripts.token_fusion import evaluate as exporter
from scripts.token_fusion import train as runner
from test_token_fusion import bank, prototypes, split


@pytest.mark.parametrize("condition", CONDITIONS)
def test_exporter_matches_training_score_path_and_keeps_tail(condition):
    dataset = split(False)
    model = TokenFusion(condition, prototypes()).eval()
    model.requires_grad_(False)
    expected, _ = score_probe(model, FeatureBatches([dataset], "concat", "cpu", training=False,
                                                    seed=1234, batch_size=4))
    # Also verify the production132-sized path including its incomplete tail.
    large = {"wavlm_layers": np.repeat(dataset.wavlm.numpy(), 24, axis=0),
             "clap_projected": np.repeat(dataset.clap.numpy(), 24, axis=0)}
    actual = exporter.score_features(model, large, "cpu")
    assert actual.shape == (144,) and np.isfinite(actual).all()
    arrays = {"wavlm_layers": dataset.wavlm.numpy(), "clap_projected": dataset.clap.numpy()}
    scored = exporter.score_features(model, arrays, "cpu")
    expected, _ = score_probe(model, FeatureBatches([dataset], "concat", "cpu", training=False, seed=1234))
    assert np.array_equal(scored, expected.numpy())


def test_verified_run_rejects_tampering_and_reloads_fixed_checkpoint(tmp_path):
    sources = bank()
    run = tmp_path / "TF0/f1/1234"
    result = runner.fit_run(sources, run, "TF0", 1234, prototypes(), {"code_sha256": {}}, "cpu",
                            logging.getLogger("test"), max_epochs=1)
    expected = {**result, "canonical_sources": {d: dict(zip(sources[d, "dev"].ids,
                  sources[d, "dev"].labels.tolist())) for d in runner.SOURCES}}
    config, completion = exporter.verified_training_run(run, expected)
    model = exporter.load_detector(run, "cpu")
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert config["model"]["condition"] == "TF0" and completion["best_epoch"] == 1
    path = run / "source_calibration.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        exporter.verified_training_run(run, expected)


def test_evaluation_copy_uses_common_contract_without_overwriting_training(tmp_path, monkeypatch):
    sources = bank()
    original = tmp_path / "training/TF0/f1/1234"
    result = runner.fit_run(sources, original, "TF0", 1234, prototypes(), {"code_sha256": {}}, "cpu",
                            logging.getLogger("copy"), max_epochs=1)
    original_hashes = {p.name: checkpoint_digest(p) for p in original.iterdir() if p.is_file()}
    run = tmp_path / "evaluation/TF0/f1/1234"
    run.mkdir(parents=True)
    for name in ("best.pt", "config.json", "training_complete.json", "source_calibration.json"):
        exporter.shutil.copy2(original / name, run / name)
    digest = checkpoint_digest(run / "best.pt")
    manifest = load_score_manifest(run, "f1", 1234, digest, create=True)
    for domain in runner.SOURCES:
        name = f"source_dev_{domain}.csv"
        exporter.shutil.copy2(original / name, run / name)
        manifest["score_sha256"][name] = checkpoint_digest(run / name)
    target = run / "target_scores.csv"
    write_score_csv(target, [("a", "speechfake", 0, -1.), ("b", "speechfake", 1, 1.)])
    manifest["score_sha256"][target.name] = checkpoint_digest(target)
    save_score_manifest(run, manifest)
    import scripts.evaluate_scores as shared
    calls = []
    def canonical(domain, part, root):
        calls.append((domain, part))
        if domain == "speechfake":
            assert calls[:3] == [(d, "dev") for d in runner.SOURCES]
            return {"a": 0, "b": 1}
        return dict(zip(sources[domain, "dev"].ids, sources[domain, "dev"].labels.tolist()))
    monkeypatch.setattr(shared, "canonical_score_labels", canonical)
    report = evaluate_run(run, "f1", seed=1234, source_data_root=tmp_path,
                          target_data_root=tmp_path, bootstrap_resamples=5)
    assert report["target"]["eer"] == 0 and report["target"]["auroc"] == 1
    assert report["source_dev"]["macro_eer"] == pytest.approx(result["source_dev"]["macro_eer"], abs=1e-14)
    assert {p.name: checkpoint_digest(p) for p in original.iterdir() if p.is_file()} == original_hashes
    with pytest.raises(FileExistsError):
        evaluate_run(run, "f1", seed=1234, source_data_root=tmp_path, target_data_root=tmp_path)


def test_comparison_requires_every_seed_not_best_test_seed():
    rows = [{"condition": c, "seed": s, "eer": .1+i*.01, "auroc": .9-i*.01}
            for c in CONDITIONS for i, s in enumerate(runner.SEEDS)]
    summary = exporter.target_summary(rows)
    for row in summary["three_seed_summary"]:
        assert row["eer_mean"] == pytest.approx(.11)
        assert row["eer_std"] == pytest.approx(.01)
    assert summary["selected_global_winner"] is False
    assert summary["exploratory_after_prior_target_exposure"] is True
    with pytest.raises(ValueError, match="All three seeds"):
        exporter.target_summary(rows[:-1])


def test_incomplete_training_queue_rejected_before_any_data_read(tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"fits": []}))
    (tmp_path / "complete.json").write_text(json.dumps({"verified_fits": 12,
        "summary_sha256": checkpoint_digest(summary)}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete queue must stop before opening data")
    monkeypatch.setattr(exporter, "canonical_score_labels", forbidden)
    with pytest.raises(ValueError, match="All twelve"):
        exporter.prepare_runs(tmp_path, tmp_path / "out", tmp_path, "cpu")


def test_nonfinite_scores_stop_instead_of_skipping_audio():
    def invalid(batch):
        return {"logit": torch.full((len(batch["wavlm"]),), float("nan"))}
    with pytest.raises(ValueError, match="Nonfinite spoof scores"):
        exporter.score_features(invalid, {"wavlm_layers": np.zeros((3,13,768), np.float32),
            "clap_projected": np.zeros((3,512), np.float32)}, "cpu")
