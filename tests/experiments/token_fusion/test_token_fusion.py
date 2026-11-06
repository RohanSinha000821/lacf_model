"""CUDA-hidden synthetic checks for isolated full-source F1 token fusion."""

import json
import logging

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from audio_deepfake_detection.protocol import checkpoint_digest
from audio_deepfake_detection.token_fusion.model import AttentionBlock, CONDITIONS, PARAMETERS, TokenFusion
from scripts.token_fusion import train as runner


def prototypes():
    return unit_vectors(torch.randn(8, 512))


def batch():
    return {"wavlm": torch.randn(4, 13, 768), "clap": torch.randn(4, 512),
            "labels": torch.tensor([0, 1, 0, 1])}


def split(training):
    n = 6
    labels = torch.arange(n) % 2
    rows = torch.arange(n * 3).reshape(n, 3) if training else torch.arange(n).reshape(n, 1)
    return FeatureSplit([(f"/{i}.wav", int(labels[i])) for i in range(n)],
                        [f"{n-i}.wav" for i in range(n)], labels, rows,
                        torch.randn(rows.numel(), 13, 768), torch.randn(rows.numel(), 512))


def bank():
    return {(d, s): split(s == "train") for d in runner.SOURCES for s in ("train", "dev")}


@pytest.mark.parametrize("condition", CONDITIONS)
def test_dimensions_precision_inventory_and_all_trainable_gradients(condition):
    proto, inputs = prototypes(), batch()
    model = TokenFusion(condition, proto)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(inputs)
        losses = model.objective(output, inputs["labels"])
    assert output["features"].shape == (4, 1024 if condition == "TF0" else 512)
    assert output["features"].dtype == losses["total"].dtype == torch.float32
    assert output["logit"].shape == (4,)
    losses["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.configuration()["trainable_parameters"] == PARAMETERS[condition]
    assert not model.configuration()["foundation_encoders_loaded"]
    assert not any(n.startswith(("wavlm.", "clap.")) for n, _ in model.named_parameters())
    assert losses["bonafide"].item() == 0
    if condition == "TF3":
        assert torch.equal(proto, model.prototypes) and model.prototypes.grad is None
        assert output["p_f"].dtype == torch.float32
        torch.testing.assert_close(output["p_f"].sum(-1), torch.ones(4))
        torch.testing.assert_close(output["v_f"].norm(dim=-1), torch.ones(4))
        expected = losses["detection"] + .25 * losses["semantic"]
    else:
        assert "p_f" not in output and not hasattr(model, "semantic_projection")
        assert losses["semantic"].item() == 0
        expected = losses["detection"]
    torch.testing.assert_close(losses["total"], expected, atol=0, rtol=0)


def test_tokens_have_correct_ids_cls_and_shared_adapter():
    model = TokenFusion("TF2", prototypes()).eval()
    inputs = batch()
    tokens = model.tokens(inputs)
    assert tokens.shape == (4, 15, 512)
    torch.testing.assert_close(tokens[:, 0], model.cls[:, 0].expand(4, -1))
    expected_w = model.token_norm(model.adapter(inputs["wavlm"])) + model.layer_ids + model.type_ids[:, :1]
    torch.testing.assert_close(tokens[:, 1:14], expected_w)
    expected_c = model.token_norm(unit_vectors(unit_vectors(inputs["clap"]))) + model.type_ids[:, 1]
    torch.testing.assert_close(tokens[:, 14], expected_c)
    assert sum(p.numel() for p in model.adapter.parameters()) == 657920
    assert len(model.blocks) == 2
    assert model.blocks[0].qkv.weight.data_ptr() != model.blocks[1].qkv.weight.data_ptr()
    assert not torch.equal(model.blocks[0].qkv.weight, model.blocks[1].qkv.weight)


def test_tf1_never_reads_clap_and_has14_tokens():
    model = TokenFusion("TF1", prototypes()).eval()
    inputs = batch()
    del inputs["clap"]
    assert model.tokens(inputs).shape == (4, 14, 512)
    assert model(inputs)["logit"].shape == (4,)


@pytest.mark.parametrize("condition", ["TF0", "TF2", "TF3"])
def test_both_feature_branches_can_affect_detection(condition):
    model = TokenFusion(condition, prototypes()).eval()
    inputs = batch()
    inputs["wavlm"].requires_grad_()
    inputs["clap"].requires_grad_()
    model(inputs)["logit"].sum().backward()
    for name in ("wavlm", "clap"):
        assert torch.isfinite(inputs[name].grad).all()
        assert inputs[name].grad.abs().sum() > 0
    assert (inputs["wavlm"].grad.abs().sum((0, 2)) > 0).all()


def test_tf2_tf3_share_starting_scores_common_weights_and_dropout_rng():
    proto, inputs = prototypes(), batch()
    models = []
    states = []
    for condition in ("TF2", "TF3"):
        torch.manual_seed(1234)
        models.append(TokenFusion(condition, proto))
        states.append(torch.get_rng_state())
    assert torch.equal(*states)
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[name])
    results = []
    for model in models:
        torch.set_rng_state(states[0])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            results.append((model(inputs)["logit"], torch.get_rng_state()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_attention_softmax_is_fp32_and_matches_manual_formula():
    q, k = torch.randn(3, 8, 15, 64).bfloat16(), torch.randn(3, 8, 15, 64).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = AttentionBlock.probabilities(q, k)
    expected = (q.float() @ k.float().transpose(-2, -1) / 8).softmax(-1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual.sum(-1), torch.ones(3, 8, 15))


def test_frozen_text_does_not_enter_detection_head_directly():
    model = TokenFusion("TF3", prototypes()).eval()
    inputs = batch()
    before = model(inputs)
    with torch.no_grad():
        model.semantic_projection.weight.mul_(-1)
    after = model(inputs)
    assert torch.equal(before["logit"], after["logit"])
    assert not torch.equal(before["p_f"], after["p_f"])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_fresh_checkpoint_exactly_reloads(condition, tmp_path):
    p, inputs = prototypes(), batch()
    first = TokenFusion(condition, p).eval()
    expected = first(inputs)["logit"]
    path = tmp_path / "state.pt"
    torch.save(first.state_dict(), path)
    second = TokenFusion(condition, p).eval()
    second.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(expected, second(inputs)["logit"])


def test_f1_loader_opens_only_six_source_partitions(tmp_path, monkeypatch):
    for name in ("config.json", "execution_accelerated.json"):
        (tmp_path / name).write_text("{}")
    marker = {"step": 2, "config_sha256": checkpoint_digest(tmp_path / "config.json"),
              "accelerated_execution_sha256": checkpoint_digest(tmp_path / "execution_accelerated.json"),
              "splits": {f"{d}/{s}": {} for d in runner.SOURCES for s in ("train", "dev")}}
    marker["splits"]["speechfake/train"] = {"must_not_open": True}
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    calls = []
    def load(root, domain, part, descriptor, data_root, **kwargs):
        assert domain in runner.SOURCES and domain != "speechfake"
        calls.append((domain, part))
        return split(part == "train")
    monkeypatch.setattr(runner, "load_split", load)
    result = runner.load_f1_bank(tmp_path, tmp_path, logging.getLogger("test"))
    assert len(calls) == len(result) == 6
    result["speechfake", "dev"] = split(False)
    with pytest.raises(ValueError, match="Only complete F1"):
        runner.assert_bank(result)


def test_source_sampler_balances_domain_and_class_and_keeps_incomplete_batch():
    sources = bank()
    loader = FeatureBatches([sources[d, "train"] for d in runner.SOURCES], "concat", "cpu", training=True, seed=1234, batch_size=7)
    weights = loader.sampler.weights
    for i in range(3):
        for label in (0, 1):
            labels = sources[runner.SOURCES[i], "train"].labels
            torch.testing.assert_close(weights[i*6:(i+1)*6][labels == label].sum(), torch.tensor(1., dtype=torch.float64))
    assert [len(b["labels"]) for b in loader] == [7, 7, 4]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_real_full_source_fit_export_calibration_completion_and_no_overwrite(condition, tmp_path):
    sources = bank()
    run = tmp_path / condition / "f1" / "1234"
    result = runner.fit_run(sources, run, condition, 1234, prototypes(), {"code_sha256": {}}, "cpu",
                            logging.getLogger("toy"), max_epochs=2)
    assert result["scope"] == runner.SCOPE
    assert set(result["source_dev"]["by_domain"]) == set(runner.SOURCES)
    completion = runner.verify_fit(run, sources)
    assert completion["selected_checkpoint_reloaded_and_source_verified"]
    assert not list(run.glob("*target*"))
    with pytest.raises(FileExistsError):
        runner.fit_run(sources, run, condition, 1234, prototypes(), {"code_sha256": {}}, "cpu", logging.getLogger("toy"), max_epochs=1)
    (run / "source_dev_asv2019.csv").write_text("changed")
    with pytest.raises(ValueError, match="Score file"):
        runner.verify_fit(run, sources)


def test_ties_keep_first_checkpoint_and_stop_after_patience(tmp_path, monkeypatch):
    sources = bank()
    def fixed(*args, **kwargs):
        scores = {d: sources[d, "dev"].labels.float() * 2 - 1 for d in runner.SOURCES}
        return scores, {"by_domain": {d: {"eer": 0., "auroc": 1.} for d in runner.SOURCES},
                        "macro_eer": 0., "macro_auroc": 1.}
    monkeypatch.setattr(runner, "score_sources", fixed)
    result = runner.fit_run(sources, tmp_path / "TF0/f1/1234", "TF0", 1234, prototypes(), {"code_sha256": {}},
                            "cpu", logging.getLogger("tie"), max_epochs=5, patience=2)
    assert result["best_epoch"] == 1 and result["final_epoch"] == 3


def test_source_guard_mutation_stops_before_run_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "code.py").write_text("before")
    provenance = {"code_sha256": {"code.py": checkpoint_digest(tmp_path / "code.py")}}
    (tmp_path / "code.py").write_text("after")
    run = tmp_path / "TF0/f1/1234"
    with pytest.raises(ValueError, match="Execution source changed"):
        runner.fit_run(bank(), run, "TF0", 1234, prototypes(), provenance, "cpu", logging.getLogger("guard"))
    assert not run.exists()


def test_summary_uses_three_seed_sd_and_never_claims_inner_or_target_metrics():
    rows = [{"condition": c, "seed": seed, "source_dev": {"macro_eer": .1 + index*.01, "macro_auroc": .9-index*.01}}
            for c in CONDITIONS for index, seed in enumerate(runner.SEEDS)]
    summary = runner.summary_for(rows)
    assert len(summary["three_seed_summary"]) == 4
    assert summary["completed_fits"] == 12 and summary["expected_fits"] == 12
    assert summary["target_access"] is False and summary["selected_global_winner"] is False
    assert "not inner-held-domain" in summary["scope"]
    for row in summary["three_seed_summary"]:
        assert row["eer_mean"] == pytest.approx(.11)
        assert row["eer_std"] == pytest.approx(.01)


def test_bad_condition_shape_and_unapproved_seed_rejected(tmp_path):
    with pytest.raises(ValueError):
        TokenFusion("unknown", prototypes())
    with pytest.raises(ValueError):
        TokenFusion("TF3", torch.ones(8, 512))
    inputs = batch()
    inputs["wavlm"] = inputs["wavlm"][:, :12]
    with pytest.raises(ValueError):
        TokenFusion("TF2", prototypes())(inputs)
    with pytest.raises(ValueError, match="condition/seed"):
        runner.fit_run(bank(), tmp_path / "TF0/f1/999", "TF0", 999, prototypes(), {"code_sha256": {}},
                       "cpu", logging.getLogger("invalid"))
