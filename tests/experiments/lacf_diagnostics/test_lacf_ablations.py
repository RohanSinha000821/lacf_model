"""Synthetic CPU checks for the plan's Step4 relation and loss ablations."""

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from audio_deepfake_detection.lacf.ablations import ALIASES, VARIANTS, RelationAblation, feature_inputs
from audio_deepfake_detection.lacf.model import CONCEPTS, TEMPLATES, LACF, unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.lacf import train_ablations, train_probes


def prototypes():
    return unit_vectors(torch.randn(8, 512))


@pytest.mark.parametrize("variant,dimension,count", [
    ("A4", 8, 1777), ("A5", 8, 659697), ("A6", 16, 660225),
    ("A7", 32, 661281), ("A8", 35, 661479), ("A9", 35, 661479), ("A9_no_sem", 35, 661479),
])
def test_exact_components_dimensions_parameters_and_frozen_inputs(variant, dimension, count):
    model = RelationAblation(variant, prototypes())
    batch = {"wavlm": torch.randn(4, 768), "clap": torch.randn(4, 512), "labels": torch.tensor([0, 1, 0, 1])}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        losses = model.objective(output, batch["labels"])
    assert output["features"].shape == (4, dimension)
    assert output["features"].dtype == losses["total"].dtype == torch.float32
    assert model.configuration()["trainable_parameters"] == count
    assert not model.prototypes.requires_grad and not hasattr(model, "wavlm") and not hasattr(model, "clap")
    losses["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not batch["wavlm"].requires_grad and not batch["clap"].requires_grad
    if variant == "A4":
        assert model.adapter is None and "p_w" not in output and feature_inputs(variant) == "clap"
        assert losses["semantic"].item() == losses["bonafide"].item() == 0
    if variant == "A5":
        assert "p_c" not in output and feature_inputs(variant) == "last"
    assert train_ablations.EXPECTED_FITS == 189 and ALIASES == {"A9_no_bf": "A8"}


@pytest.mark.parametrize("training", [False, True])
def test_a9_is_bit_exact_to_primary_backend_including_gradients(training):
    w, c = nn.Identity(), nn.Identity()
    w.config, c.config = SimpleNamespace(hidden_size=768), SimpleNamespace(projection_dim=512)
    primary = LACF(w, c, prototypes()).train(training)
    cached = RelationAblation("A9", primary.prototypes).train(training)
    cached.adapter.load_state_dict(primary.adapter.state_dict())
    cached.classifier.load_state_dict(primary.classifier.state_dict())
    batch = {"wavlm": torch.randn(5, 768), "clap": torch.randn(5, 512), "labels": torch.tensor([0, 1, 0, 1, 1])}
    torch.manual_seed(1234)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        left = primary.classify_audio({"z_w": batch["wavlm"], "v_c": unit_vectors(unit_vectors(batch["clap"]))})
        loss_left = primary.objective(left, batch["labels"])
    torch.manual_seed(1234)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        right = cached(batch)
        loss_right = cached.objective(right, batch["labels"])
    for key in ("v_w", "v_c", "p_w", "p_c", "features", "logit"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for key in loss_left:
        torch.testing.assert_close(loss_left[key], loss_right[key], rtol=0, atol=0)
    loss_left["total"].backward()
    loss_right["total"].backward()
    for (_, p), (_, q) in zip(primary.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_loss_removals_and_all_spoof_microbatch_follow_primary_equations():
    full = RelationAblation("A9", prototypes()).eval()
    batch = {"wavlm": torch.randn(4, 768), "clap": torch.randn(4, 512)}
    labels = torch.tensor([0, 1, 1, 0])
    output = full(batch)
    losses = full.objective(output, labels)
    no_bf = RelationAblation("A8", full.prototypes)
    no_sem = RelationAblation("A9_no_sem", full.prototypes)
    torch.testing.assert_close(no_bf.objective(output, labels)["total"], losses["detection"] + .25 * losses["semantic"])
    torch.testing.assert_close(no_sem.objective(output, labels)["total"], losses["detection"] + .1 * losses["bonafide"])
    assert full.objective(output, torch.ones(4))["bonafide"].item() == 0
    with pytest.raises(ValueError, match="duplicate"):
        RelationAblation("A9_no_bf", full.prototypes)


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_shared_fit_loop_reloads_source_ablation_and_refuses_overwrite(tmp_path, monkeypatch, variant):
    def split(training):
        count = 4
        aliases = torch.arange(count * 3).reshape(count, 3) if training else torch.arange(count).reshape(count, 1)
        return FeatureSplit([(f"/{i}.wav", i % 2) for i in range(count)], [f"{3-i}.wav" for i in range(count)],
                            torch.arange(count) % 2, aliases, torch.randn(aliases.numel(), 768),
                            torch.randn(aliases.numel(), 512))
    bank = {("asv5", "train"): split(True), ("cfad", "train"): split(True), ("asv2019", "dev"): split(False)}
    proto = prototypes()
    monkeypatch.setattr(train_probes, "MAX_EPOCHS", 2)
    inputs = feature_inputs(variant)
    run = tmp_path / variant / "inner_asv2019/f1/1234"
    options = dict(fold="f1", held="asv2019", inputs=variant, head="relation", seed=1234,
                   provenance={"scope": "synthetic CPU"}, device="cpu", logger=logging.getLogger(variant),
                   model_factory=lambda name, head: RelationAblation(name, proto), stage=4,
                   feature_inputs=inputs, last_layer_only=inputs != "clap")
    result = train_probes.fit_probe(bank, run, **options)
    assert result["final_epoch"] == 2
    config = json.loads((run / "config.json").read_text())
    assert config["stage"] == 4 and config["ablation"]["variant"] == variant
    assert config["fit_domains"] == ["asv5", "cfad"] and config["foundation_encoders_loaded"] is False
    train_probes.verify_probe_completion(run, "f1", 1234, expected_stage=4)
    with pytest.raises(ValueError, match="verification"):
        train_probes.verify_probe_completion(run, "f1", 1234)
    with pytest.raises(FileExistsError):
        train_probes.fit_probe(bank, run, **options)


def test_prototype_extraction_ignores_old_learned_weights_and_detects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(train_ablations, "ROOT", tmp_path)
    proto = prototypes()
    evidence = {"checkpoint_sha256": {}}
    for i, fold in enumerate(("f1", "f2")):
        path = tmp_path / "outputs/lacf10s_optimized_20261009" / fold / "1234/best.pt"
        path.parent.mkdir(parents=True)
        torch.save({"config": {"components": {"concepts": list(CONCEPTS), "templates": list(TEMPLATES), "temperature": .07}},
                    "model_state_dict": {"prototypes": proto, "old_learned_weights": torch.tensor([i + 987.])}}, path)
        evidence["checkpoint_sha256"]["lacf_" + fold] = checkpoint_digest(path)
    static = tmp_path / "outputs/lacf_diagnostics_step1/static.json"
    static.parent.mkdir(parents=True); static.write_text(json.dumps(evidence))
    actual, provenance = train_ablations.fixed_prototypes()
    torch.testing.assert_close(actual, proto, rtol=0, atol=0)
    assert provenance["learned_checkpoint_parameters_loaded"] is False
    with (path).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        train_ablations.fixed_prototypes()
