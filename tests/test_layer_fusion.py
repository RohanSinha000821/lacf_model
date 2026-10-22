"""CUDA-hidden synthetic checks for the separate frozen C0–C4 comparison."""

import json
import logging

import pytest
import torch

from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss, unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from audio_deepfake_detection.layer_fusion.diagnostics import source_diagnostics
from audio_deepfake_detection.layer_fusion.model import (
    CONDITIONS, GROUPS, PARAMETERS, LayerFusion, LayerFusionDetector,
)
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.lacf import train_probes
from scripts.layer_fusion.train import EXPECTED_FITS, STAGE, fit_condition


def prototype_bank():
    return unit_vectors(torch.randn(8, 512))


def audio_batch():
    return {"wavlm": torch.randn(6, 13, 768), "clap": torch.randn(6, 512),
            "labels": torch.tensor([0, 1, 0, 1, 0, 1])}


def split(training):
    count = 6
    rows = torch.arange(count * 3).reshape(count, 3) if training else torch.arange(count).reshape(count, 1)
    return FeatureSplit([(f"/{i}.wav", i % 2) for i in range(count)], [f"{count-i}.wav" for i in range(count)],
                        torch.arange(count) % 2, rows, torch.randn(rows.numel(), 13, 768),
                        torch.randn(rows.numel(), 512))


def source_bank():
    return {(domain, part): split(part == "train") for domain in FOLDS["f1"]["sources"] for part in ("train", "dev")}


@pytest.mark.parametrize("condition", CONDITIONS)
def test_primary_features_loss_counts_and_frozen_foundation(condition):
    proto, batch = prototype_bank(), audio_batch()
    model = LayerFusionDetector(condition, proto)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        losses = model.objective(output, batch["labels"])
    assert output["features"].shape == (6, 35)
    for key in ("p_w", "p_c", "v_w", "v_c", "features"):
        assert output[key].dtype == torch.float32 and torch.isfinite(output[key]).all()
    assert losses["total"].dtype == torch.float32
    torch.testing.assert_close(losses["total"], losses["detection"], atol=0, rtol=0)
    assert losses["semantic"].item() == losses["bonafide"].item() == 0
    losses["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not model.backend.prototypes.requires_grad and model.backend.prototypes.grad is None
    assert torch.equal(model.backend.prototypes, proto)
    assert not hasattr(model, "wavlm") and not hasattr(model, "clap")
    config = model.configuration()
    assert config["trainable_parameters"] == PARAMETERS[condition]
    assert config["backend"]["trainable_parameters"] == 661479
    assert config["backend"]["classifier"] == [35, 64, 16, 1]
    assert not config["backend"]["semantic_loss"] and not config["backend"]["bonafide_loss"]
    assert not config["foundation_encoders_loaded"]


def test_layer_rules_uniform_initialization_and_expert_capacity_control():
    x = torch.randn(4, 13, 768)
    assert GROUPS == ((0, 4), (4, 8), (8, 13))
    torch.testing.assert_close(LayerFusion("C0")(x)[0], x[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(LayerFusion("C1")(x)[0], x.mean(1), atol=0, rtol=0)
    torch.testing.assert_close(LayerFusion("C2")(x)[0], x.mean(1), atol=1e-6, rtol=1e-6)
    c3 = LayerFusion("C3")
    actual, weights = c3(x)
    expected = torch.stack([expert(x[:, start:end].mean(1)) for expert, (start, end) in zip(c3.experts, GROUPS)], 1).mean(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, torch.full((4, 3), 1 / 3))
    assert sum(p.numel() for p in c3.experts[0].parameters()) == 100672
    c4 = LayerFusion("C4")
    assert PARAMETERS["C4"] - PARAMETERS["C3"] == 3840
    torch.testing.assert_close(c4.routing_weights(x), weights, atol=0, rtol=0)
    # Input dependence is possible once the initially uniform router learns.
    with torch.no_grad():
        c4.router[-1].weight[0, 0] = 1
    assert not torch.equal(c4.routing_weights(x[:1]), c4.routing_weights(x[1:2]))
    with pytest.raises(ValueError):
        LayerFusion("unknown")


def test_common_initialization_rng_and_experts_are_matched_between_conditions():
    proto = prototype_bank()
    models, rng = [], []
    for condition in CONDITIONS:
        torch.manual_seed(1234)
        models.append(LayerFusionDetector(condition, proto))
        rng.append(torch.get_rng_state())
    for model, state in zip(models[1:], rng[1:]):
        assert torch.equal(state, rng[0])
        for name, value in models[0].backend.state_dict().items():
            assert torch.equal(value, model.backend.state_dict()[name])
    for name, value in models[3].layer_fusion.experts.state_dict().items():
        assert torch.equal(value, models[4].layer_fusion.experts.state_dict()[name])
    batch = audio_batch()
    for model in models[3:]:
        model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(models[3](batch)["logit"], models[4](batch)["logit"])


@pytest.mark.parametrize("training", [False, True])
def test_c0_exact_primary_backend_behavior_except_authorized_auxiliary_loss_removal(training):
    proto, batch = prototype_bank(), audio_batch()
    torch.manual_seed(1234)
    candidate = LayerFusionDetector("C0", proto).train(training)
    torch.manual_seed(1234)
    reference = RelationAblation("A9", proto).train(training)
    reference.objective = LACFLoss(ComponentConfig(semantic_loss=False, bonafide_loss=False,
                                                  semantic_weight=0, bonafide_weight=0))
    torch.manual_seed(42)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        left = candidate(batch)
        loss_left = candidate.objective(left, batch["labels"])["total"]
    torch.manual_seed(42)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        right = reference({"wavlm": batch["wavlm"][:, -1], "clap": batch["clap"]})
        loss_right = reference.objective(right, batch["labels"])["total"]
    for key in ("v_w", "v_c", "p_w", "p_c", "features", "logit"):
        assert torch.equal(left[key], right[key])
    assert torch.equal(loss_left, loss_right)
    loss_left.backward()
    loss_right.backward()
    for (_, p), (_, q) in zip(candidate.backend.named_parameters(), reference.named_parameters()):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_fresh_inner_fit_reload_export_routing_and_no_overwrite(tmp_path, monkeypatch, condition):
    bank, proto = source_bank(), prototype_bank()
    monkeypatch.setattr(train_probes, "MAX_EPOCHS", 2)
    run = tmp_path / condition / "inner_asv2019/f1/1234"
    options = dict(condition=condition, fold="f1", held="asv2019", seed=1234,
                   provenance={"scope": "synthetic CPU"}, device="cpu", logger=logging.getLogger(condition),
                   factory=lambda condition, head: LayerFusionDetector(condition, proto))
    result = fit_condition(bank, run, **options)
    assert result["final_epoch"] == 2
    train_probes.verify_probe_completion(run, "f1", 1234, expected_stage=STAGE)
    config = json.loads((run / "config.json").read_text())
    assert config["stage"] == STAGE and config["layer_fusion"]["condition"] == condition
    assert config["fit_domains"] == ["asv5", "cfad"]
    assert config["feature_inputs"] == "concat" and not config["last_layer_only"]
    assert config["loader_workers"] == 0 and not config["model_training_resume"]
    report = json.loads((run / "source_diagnostics.json").read_text())
    assert result["source_diagnostics_sha256"] == checkpoint_digest(run / "source_diagnostics.json")
    assert report["checkpoint_sha256"] == checkpoint_digest(run / "best.pt")
    if condition in ("C3", "C4"):
        assert set(report["domains"]) == set(FOLDS["f1"]["sources"])
        for domain in report["domains"].values():
            assert domain["all"]["count"] == 6
            assert domain["bonafide"]["count"] == domain["spoof"]["count"] == 3
            assert sum(domain["all"]["mean_weights"]) == pytest.approx(1)
            assert sum(domain["all"]["top_expert_counts"]) == 6
    else:
        assert report["domains"] == {}
    with pytest.raises(ValueError, match="verification"):
        train_probes.verify_probe_completion(run, "f1", 1234, expected_stage=4)
    with pytest.raises(FileExistsError):
        fit_condition(bank, run, **options)


def test_real_fit_path_pairs_sampling_views_and_initial_dropout(tmp_path, monkeypatch):
    bank, proto, observed = source_bank(), prototype_bank(), []
    monkeypatch.setattr(train_probes, "MAX_EPOCHS", 1)
    original = train_probes.train_epoch
    def capture(model, loader, optimizer, device, **kwargs):
        batches = [{key: value.clone() for key, value in batch.items()} for batch in loader]
        state = torch.get_rng_state()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            initial = model(batches[0])["logit"].detach().clone()
        torch.set_rng_state(state)
        observed.append((batches, initial, state))
        return original(model, batches, optimizer, device, **kwargs)
    monkeypatch.setattr(train_probes, "train_epoch", capture)
    for condition in ("C3", "C4"):
        fit_condition(bank, tmp_path / condition / "f1/1234", condition=condition, fold="f1", held="asv2019", seed=1234,
                      provenance={"scope": "synthetic CPU"}, device="cpu", logger=logging.getLogger(condition),
                      factory=lambda condition, head: LayerFusionDetector(condition, proto))
    for a, b in zip(observed[0][0], observed[1][0]):
        assert all(torch.equal(a[key], b[key]) for key in a)
    assert torch.equal(observed[0][1], observed[1][1])
    assert torch.equal(observed[0][2], observed[1][2])


def test_f4_targets_and_outer_target_validation_rejected(tmp_path):
    model = LayerFusionDetector("C4", prototype_bank())
    with pytest.raises(ValueError, match="F1–F3"):
        source_diagnostics(model, {}, "f4", 1234, "cpu")
    with pytest.raises(ValueError, match="held-in-source"):
        train_probes.fit_probe({}, tmp_path / "forbidden", fold="f1", held="speechfake", inputs="C0",
                               head="relation", seed=1234, provenance={}, device="cpu", logger=logging.getLogger())
    assert not (tmp_path / "forbidden").exists()
    assert EXPECTED_FITS == 135


def test_source_routing_reports_uniform_weights_entropy_and_tie_behavior():
    bank = source_bank()
    model = LayerFusionDetector("C4", prototype_bank())
    report = source_diagnostics(model, bank, "f1", 1234, "cpu")
    for groups in report["domains"].values():
        for group in groups.values():
            assert group["mean_weights"] == pytest.approx([1 / 3] * 3)
            assert group["mean_entropy_nats"] == pytest.approx(float(torch.tensor(3.).log()))
            assert group["std_weights"] == pytest.approx([0.] * 3, abs=1e-6)
            assert group["top_expert_counts"] == [group["count"], 0, 0]


def test_nonuniform_routing_statistics_with_unequal_class_counts():
    import numpy as np
    bank = source_bank()
    for (domain, part), dataset in bank.items():
        if part == "dev":
            dataset.labels[:] = torch.tensor([0, 1, 1, 1, 1, 1])
    model = LayerFusionDetector("C4", prototype_bank())
    with torch.no_grad():
        model.layer_fusion.router[-1].weight.normal_(0, .2)
        model.layer_fusion.router[-1].bias.copy_(torch.tensor([.5, -.2, .1]))
    report = source_diagnostics(model, bank, "f1", 1234, "cpu")
    for domain in FOLDS["f1"]["sources"]:
        dataset = bank[domain, "dev"]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            weights = model.layer_fusion.routing_weights(dataset.wavlm).numpy()
        for name, mask in (("all", np.ones(6, dtype=bool)), ("bonafide", dataset.labels.numpy() == 0),
                           ("spoof", dataset.labels.numpy() == 1)):
            selected = weights[mask].astype(np.float64)
            actual = report["domains"][domain][name]
            assert actual["count"] == len(selected)
            np.testing.assert_allclose(actual["mean_weights"], selected.mean(0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual["std_weights"], selected.std(0), rtol=0, atol=1e-7)
            assert actual["top_expert_counts"] == np.bincount(selected.argmax(1), minlength=3).tolist()
            entropy = -(selected * np.log(np.clip(selected, 1e-8, None))).sum(1).mean()
            assert actual["mean_entropy_nats"] == pytest.approx(entropy, abs=1e-6)


@pytest.mark.parametrize("condition", ("C3", "C4"))
def test_expert_and_router_linear_work_uses_bf16(condition):
    model = LayerFusionDetector(condition, prototype_bank())
    seen = []
    hooks = [module.register_forward_hook(lambda module, args, result: seen.append(result.dtype))
             for module in model.layer_fusion.modules() if isinstance(module, torch.nn.Linear)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(audio_batch())
    for hook in hooks:
        hook.remove()
    assert seen and set(seen) == {torch.bfloat16}
    assert result["routing"].dtype == result["features"].dtype == torch.float32


def test_tiny_cpu_queue_exercises_all_rotations_seeds_and_completion(tmp_path, monkeypatch):
    from scripts.layer_fusion import train
    bank, proto = source_bank(), prototype_bank()
    for domain in ("speechfake",):
        for part in ("train", "dev"):
            bank[domain, part] = split(part == "train")
    monkeypatch.setattr(train, "CONDITIONS", ("C0",))
    monkeypatch.setattr(train, "EXPECTED_FITS", 27)
    monkeypatch.setattr(train_probes, "MAX_EPOCHS", 1)
    train.run_queue(bank, tmp_path, lambda condition, head: LayerFusionDetector(condition, proto),
                    {"scope": "synthetic CPU only", "code_sha256": {}}, "cpu", logging.getLogger("tiny_queue"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    completion = json.loads((tmp_path / "complete.json").read_text())
    assert summary["complete"] and summary["completed_fits"] == 27
    assert len(summary["three_seed_summary"]) == 3
    assert {(row["fold"], row["seed"]) for row in summary["seed_macros"]} == {
        (fold, seed) for fold in ("f1", "f2", "f3") for seed in (1234, 1235, 1236)}
    assert completion["summary_sha256"] == checkpoint_digest(tmp_path / "summary.json")
    assert not (tmp_path / "summary.json.tmp").exists()
    # The same production queue guard must stop before creating any fit.
    bad = tmp_path / "bad_code"
    bad.mkdir()
    with pytest.raises(ValueError, match="implementation changed"):
        train.run_queue(bank, bad, None, {"code_sha256": {"scripts/layer_fusion/train.py": "changed"}},
                        "cpu", logging.getLogger("changed_code"))
    assert not (bad / "complete.json").exists() and not (bad / "C0").exists()


def test_main_refuses_unavailable_cuda_before_creating_outputs(tmp_path, monkeypatch):
    import sys
    from scripts.layer_fusion import train
    monkeypatch.setattr(sys, "argv", ["train", "--output", str(tmp_path / "not_created")])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Native CUDA BF16"):
        train.main()
    assert not (tmp_path / "not_created").exists()
