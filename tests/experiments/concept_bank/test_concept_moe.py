"""Synthetic bank metadata, unchanged MoE math and source-completion gates."""

import json
import logging
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from audio_deepfake_detection.experiments.concept_bank.model import ConceptBankMoE, bank_definition
from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusionDetector
from audio_deepfake_detection.lacf.model import CONCEPTS
from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.concept_bank import train_moe as runner


def prototypes():
    generator = torch.Generator().manual_seed(29)
    return F.normalize(torch.randn(8, 512, generator=generator), dim=-1)


def bank():
    generator = torch.Generator().manual_seed(9)
    result = {}
    for domain in runner.SOURCES:
        for split in ("train", "dev"):
            labels = torch.tensor([0, 0, 1, 1, 1, 1])
            count = len(labels); rows = count*3 if split == "train" else count
            views = torch.arange(rows).reshape(count, 3) if split == "train" else torch.arange(count)[:, None].expand(-1, 3)
            ids = [f"{domain}/{split}/{i}" for i in range(count)]
            result[domain, split] = FeatureSplit(list(zip(ids, labels.tolist())), ids, labels, views,
                torch.randn(rows, 13, 768, generator=generator), torch.randn(rows, 512, generator=generator))
    return result


def test_banks_differ_only_at_final_spoof_and_record_explicit_groups():
    a, b = bank_definition("vocoder"), bank_definition("neural_synthesis")
    assert len(a["concepts"]) == len(set(a["concepts"])) == 8
    assert a["concepts"][:7] == b["concepts"][:7] and a["concepts"][7] != b["concepts"][7]
    assert a["group_indices"] == {"bona_fide": [0, 1, 2, 3], "spoof": [4, 5, 6, 7]}
    assert a["uniform_spoof_mass"] == .5 and a["temperature"] == .07
    a["group_indices"]["spoof"].clear()
    assert bank_definition("vocoder")["group_indices"]["spoof"] == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        bank_definition("unknown")


@pytest.mark.parametrize("name", ["vocoder", "neural_synthesis"])
def test_bank_wrapper_keeps_real_c4_math_and_true_metadata(name):
    p = prototypes(); torch.manual_seed(1234)
    base = LayerFusionDetector("C4", p).eval()
    torch.manual_seed(1234)
    model = ConceptBankMoE(name, p).eval()
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    batch = {"wavlm": torch.randn(5, 13, 768), "clap": torch.randn(5, 512)}
    with torch.inference_mode():
        expected, actual = base(batch), model(batch)
    assert torch.equal(expected["logit"], actual["logit"])
    assert torch.equal(expected["features"], actual["features"])
    assert actual["features"].shape == (5, 35)
    config = model.configuration()
    assert config["trainable_parameters"] == 967338
    assert config["backend"]["concepts"] == bank_definition(name)["concepts"]
    assert config["backend"]["group_indices"]["spoof"] == [4, 5, 6, 7]
    assert config["backend"]["concepts"] != list(CONCEPTS)
    assert not config["semantic_loss"] and not config["bonafide_loss"]
    assert base.configuration()["backend"]["concepts"] == list(CONCEPTS)


def test_fresh_initialization_matches_between_banks_and_no_auxiliary_split_runs():
    p = prototypes(); other = p.clone(); other[7] = -other[7]
    torch.manual_seed(1234); a = ConceptBankMoE("vocoder", p)
    torch.manual_seed(1234); b = ConceptBankMoE("neural_synthesis", other)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    for labels in [torch.zeros(5), torch.ones(5), torch.tensor([0., 1., 0., 1., 1.])]:
        logit = torch.tensor([-.7, .2, .8, -.3, .1], requires_grad=True)
        losses = a.objective({"logit": logit}, labels)
        assert torch.equal(losses["total"], F.binary_cross_entropy_with_logits(logit, labels))
        assert losses["semantic"] == losses["bonafide"] == 0
    a.backend.config = replace(a.backend.config, semantic_loss=True)
    with pytest.raises(ValueError, match="detection BCE only"):
        _ = a.objective


@pytest.mark.parametrize("name", ["vocoder", "neural_synthesis"])
def test_real_fit_reloads_records_bank_and_exports_all_source_calibration(tmp_path, name):
    p = prototypes(); source = bank(); output = tmp_path / name
    result = runner.fit(source, p, name, output, {"code_sha256": {}}, "cpu", logging.getLogger(name), max_epochs=2)
    assert result["selected_checkpoint_reloaded_and_source_verified"] and not result["target_access"]
    assert runner.verify_completion(output, name) == result
    config = json.loads((output / "config.json").read_text())
    assert config["concept_bank"] == bank_definition(name)
    assert config["model"]["backend"]["concepts"] == bank_definition(name)["concepts"]
    assert config["train_records"] == 18 and config["seed_plan"] == [1234]
    assert config["batch_size"] == 132 and config["accumulation"] == 1
    assert config["lr"] == .0003 and config["weight_decay"] == .0001 and config["scheduler"] == "constant"
    assert result["prototype_bytes_sha256"] == runner.tensor_digest(p)
    assert len(result["source_scores_sha256"]) == 3 and not list(output.glob("*target*"))
    calibration = json.loads((output / "source_calibration.json").read_text())
    assert set(calibration["raw_thresholds"]) == {"1%", "5%", "10%"}
    with pytest.raises(FileExistsError):
        runner.fit(source, p, name, output, {"code_sha256": {}}, "cpu", logging.getLogger(name), max_epochs=1)
    with pytest.raises(ValueError):
        runner.verify_completion(output, "neural_synthesis" if name == "vocoder" else "vocoder")


def test_ties_keep_earliest_and_patience_stops(tmp_path, monkeypatch):
    source = bank()
    def scores(*args, **kwargs):
        values = {d: source[d, "dev"].labels.float()*2-1 for d in runner.SOURCES}
        metrics = {"by_domain": {d: {"eer": 0., "auroc": 1.} for d in runner.SOURCES}, "macro_eer": 0., "macro_auroc": 1.}
        return values, metrics
    monkeypatch.setattr(runner, "score_sources", scores)
    result = runner.fit(source, prototypes(), "vocoder", tmp_path / "run", {"code_sha256": {}}, "cpu",
        logging.getLogger("ties"), max_epochs=8, patience=2)
    assert result["best_epoch"] == 1 and result["final_epoch"] == 3


def test_changed_input_or_wrong_source_domain_stops_before_fit(tmp_path):
    source = bank(); path = tmp_path / "bank.json"; path.write_text("initial")
    provenance = {"code_sha256": {}, "input_sha256": {str(path): checkpoint_digest(path)}}
    path.write_text("changed")
    with pytest.raises(ValueError, match="concept input changed"):
        runner.fit(source, prototypes(), "vocoder", tmp_path / "run", provenance, "cpu", logging.getLogger("input"))
    assert not (tmp_path / "run").exists()
    source['speechfake', 'dev'] = source['asv2019', 'dev']
    with pytest.raises(ValueError, match="Only complete F1"):
        runner.fit(source, prototypes(), "vocoder", tmp_path / "wrong", {"code_sha256": {}}, "cpu", logging.getLogger("source"))


def test_reload_mismatch_never_marks_completion(tmp_path, monkeypatch):
    source = bank(); original = runner.score_sources; calls = 0
    def mismatch(*args, **kwargs):
        nonlocal calls
        calls += 1
        scores, metrics = original(*args, **kwargs)
        if calls == 2:
            scores[runner.SOURCES[0]] = scores[runner.SOURCES[0]].clone()
            scores[runner.SOURCES[0]][0] += 1
        return scores, metrics
    monkeypatch.setattr(runner, "score_sources", mismatch)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="full source reload differs"):
        runner.fit(source, prototypes(), "vocoder", output, {"code_sha256": {}}, "cpu", logging.getLogger("reload"), max_epochs=1)
    assert not (output / "training_complete.json").exists()


@pytest.mark.parametrize("artifact", ["config.json", "best.pt", "source_calibration.json", "source_dev_asv5.csv", "training_complete.json"])
def test_changed_completed_artifact_rejected(tmp_path, artifact):
    output = tmp_path / "run"
    runner.fit(bank(), prototypes(), "vocoder", output, {"code_sha256": {}}, "cpu", logging.getLogger("tamper"), max_epochs=1)
    path = output / artifact
    if artifact == "training_complete.json":
        path.unlink()
    else:
        with path.open("ab") as handle: handle.write(b" ")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.verify_completion(output, "vocoder")
