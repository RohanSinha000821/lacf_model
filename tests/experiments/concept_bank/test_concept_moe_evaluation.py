"""Synthetic source gates and complete bank-specific raw/transductive evaluation."""

import hashlib
import json
import logging

import numpy as np
import pytest

from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.concept_bank import evaluate_moe as evaluator
from scripts.experiments.concept_bank import train_moe as trainer
from scripts.experiments.layer_fusion import evaluate_lacf_layer_moe as shared
from test_concept_moe import bank, prototypes


@pytest.fixture(params=["vocoder", "neural_synthesis"])
def example(tmp_path, monkeypatch, request):
    name = request.param; sources = bank()
    cache = tmp_path / "source_cache"; cache.mkdir()
    for filename, value in {"config.json": {}, "execution_accelerated.json": {"pretrained": {}},
            "complete.json": {"splits": {f"{d}/dev": {} for d in trainer.SOURCES}}}.items():
        (cache / filename).write_text(json.dumps(value))
    provenance = {"code_sha256": {}, "cache": str(cache), **{key: checkpoint_digest(cache / filename)
        for filename, key in (("config.json", "cache_config_sha256"), ("complete.json", "cache_completion_sha256"),
            ("execution_accelerated.json", "cache_execution_sha256"))}}
    run = tmp_path / "training" / name / "f1/1234"
    trainer.fit(sources, prototypes(), name, run, provenance, "cpu", logging.getLogger(name), max_epochs=1)
    targets = {f"target{i}": int(i >= 2) for i in range(6)}
    source_labels = {d: dict(zip(sources[d, "dev"].ids, sources[d, "dev"].labels.tolist())) for d in trainer.SOURCES}
    canonical = lambda domain, split, root: targets if domain == "speechfake" else source_labels[domain]
    monkeypatch.setattr(evaluator, "canonical_score_labels", canonical)
    monkeypatch.setattr(shared, "canonical_score_labels", canonical)
    monkeypatch.setattr("scripts.evaluate_scores.canonical_score_labels", canonical)
    monkeypatch.setattr(shared, "load_split", lambda root, domain, split, *args, **kwargs: sources[domain, split])
    features = tmp_path / "prior_evaluation/target_features"; features.mkdir(parents=True)
    extraction = {"provenance": {"cache_root": str(cache), "cache_completion_sha256": checkpoint_digest(cache / "complete.json")},
        "frozen_encoder_revisions": {}, "batching": {"wavlm": 32, "clap": 64, "head": 132, "chunk": 1024, "workers": 8, "prefetch": 2}}
    (features.parent / "evaluation_config.json").write_text(json.dumps(extraction))
    chunk = features / "chunk_00000"; chunk.mkdir()
    rows = []
    for i, (identifier, label) in enumerate(targets.items()):
        frames = 200000 if i % 2 else 4000
        intervals, aliases = shared.crop_intervals(frames, 16000, identifier, training=False)
        rows.append({"index": i, "id": f"speechfake/test/{identifier}", "label": label,
            "frames": frames, "rate": 16000, "intervals": [list(x) for x in intervals], "view_rows": aliases})
    (chunk / "records.json").write_text(json.dumps(rows))
    np.save(chunk / "wavlm_layers.npy", sources["asv2019", "dev"].wavlm.numpy())
    np.save(chunk / "clap_projected.npy", sources["asv2019", "dev"].clap.numpy())
    (chunk / "complete.json").write_text(json.dumps({"records": 6, "files_sha256": {
        f: checkpoint_digest(chunk / f) for f in ("records.json", "wavlm_layers.npy", "clap_projected.npy")}}))
    (features / "complete.json").write_text(json.dumps({"records": 6,
        "evaluation_config_sha256": checkpoint_digest(features.parent / "evaluation_config.json"),
        "canonical_ids_labels_sha256": hashlib.sha256(json.dumps(list(targets.items())).encode()).hexdigest(),
        "chunks": [{"path": chunk.name, "completion_sha256": checkpoint_digest(chunk / "complete.json")}]}))
    return name, run, features, tmp_path / "evaluation" / name / "f1/1234"


def test_real_bank_evaluation_preserves_calibration_and_training(example, tmp_path):
    name, run, features, output = example
    before = {p.name: checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    report = evaluator.evaluate(run, output, name, features, tmp_path, "cpu", logging.getLogger(name), bootstrap_resamples=0)
    assert report["model_family"] == "lacf_layer_moe_balanced_concepts"
    assert report["target"]["n_bona_fide"] == 2 and report["target"]["n_spoof"] == 4
    assert set(report["strict_raw_transfer"]["target_operating_points"]) == {"1%", "5%", "10%"}
    assert report["unlabeled_zscore_transfer"]["transductive"]
    assert json.loads((output / "evaluation_config.json").read_text())["concept_bank"]["name"] == name
    assert json.loads((output / "evaluation_complete.json").read_text())["no_target_tuning"]
    assert all(checkpoint_digest(run / filename) == digest for filename, digest in before.items())
    with pytest.raises(FileExistsError):
        evaluator.evaluate(run, output, name, features, tmp_path, "cpu", logging.getLogger(name))


@pytest.mark.parametrize("change", ["artifact", "wrong_bank", "source_reload"])
def test_failed_source_gate_never_opens_target(example, tmp_path, monkeypatch, change):
    name, run, features, output = example
    if change == "artifact":
        with (run / "source_calibration.json").open("a") as f:
            f.write("changed")
    elif change == "wrong_bank":
        name = "neural_synthesis" if name == "vocoder" else "vocoder"
        output = tmp_path / "evaluation" / name / "f1/1234"
    else:
        original = shared.score_probe
        def mismatch(*args, **kwargs):
            scores, metrics = original(*args, **kwargs)
            scores = scores.clone(); scores[0] += 1
            return scores, metrics
        monkeypatch.setattr(shared, "score_probe", mismatch)
    monkeypatch.setattr(evaluator, "target_chunks", lambda *args: pytest.fail("Target must remain unopened"))
    with pytest.raises(ValueError):
        evaluator.evaluate(run, output, name, features, tmp_path, "cpu", logging.getLogger(name), bootstrap_resamples=0)
    assert not output.exists()
