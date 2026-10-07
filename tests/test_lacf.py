"""Synthetic CPU fidelity tests. No downloaded checkpoints or real dataset reads."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import ClapFeatureExtractor, Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from audio_deepfake_detection.lacf import data, model, pretrained, training
from audio_deepfake_detection.protocol import FOLDS, verify_completed_run
from scripts.lacf import train


class FakeWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.projection = nn.Linear(1, 768)
        self.dropout = nn.Dropout(0.9)

    def forward(self, input_values, attention_mask, return_dict):
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(input_values.unsqueeze(-1))))

    def _get_feature_vector_attention_mask(self, length, mask):
        return mask.bool()


class FakeClap(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=512)
        self.projection = nn.Linear(2, 512)
        self.dropout = nn.Dropout(0.9)

    def get_audio_features(self, input_features):
        return self.dropout(self.projection(input_features))


def detector(config=model.ComponentConfig(), **kwargs):
    torch.manual_seed(1234)
    return model.LACF(FakeWavLM(), FakeClap(), torch.randn(8, 512), config=config, **kwargs)


def batch():
    return {"wavlm": {"input_values": torch.tensor([[1., 2., 900.], [2., 3., 4.]]),
                      "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])},
            "clap": {"input_features": torch.tensor([[0.2, 0.3], [0.4, 0.1]])},
            "labels": torch.tensor([0, 1])}


def test_exact_prompt_order_and_two_stage_prototype_normalization():
    assert model.CONCEPTS == (
        "natural human speech", "authentic human voice recording", "naturally produced human voice",
        "synthetic speech", "text-to-speech generated voice", "voice-converted speech",
        "AI-cloned human voice", "neural-vocoder generated speech")
    assert model.TEMPLATES == ("{concept}", "a recording of {concept}", "this audio contains {concept}")
    values = torch.zeros(24, 512)
    for i in range(8):
        values[3*i, i] = 2
        values[3*i+1, i+8] = 8
        values[3*i+2, i] = 4
    expected = F.normalize(F.normalize(values, dim=-1).reshape(8, 3, 512).mean(1), dim=-1)
    torch.testing.assert_close(model.text_prototypes(values), expected)
    assert not torch.allclose(expected, F.normalize(values.reshape(8, 3, 512).mean(1), dim=-1))
    with pytest.raises(ValueError):
        model.text_prototypes(torch.zeros(24, 512))


def test_primary_feature_order_dimensions_js_and_entropies():
    p = torch.tensor([[.1, .1, .1, .1, .1, .1, .1, .3]])
    q = torch.tensor([[.2, .1, .1, .1, .1, .1, .1, .2]])
    midpoint = (p + q) / 2
    js = ((p * (p / midpoint).log()).sum(-1) + (q * (q / midpoint).log()).sum(-1)) / 2
    expected = torch.cat((p, q, (p-q).abs(), p*q, js[:, None],
                          -(p*p.log()).sum(-1)[:, None], -(q*q.log()).sum(-1)[:, None]), -1)
    features = model.RelationFeatures()({"p_w": p, "p_c": q})
    assert features.shape == (1, 35)
    torch.testing.assert_close(features, expected)
    torch.testing.assert_close(model.js_divergence(p, p), torch.zeros(1))
    p = torch.tensor([[1., 0.]], requires_grad=True)
    q = torch.tensor([[0., 1.]])
    js = model.js_divergence(p, q)
    assert js.item() == pytest.approx(np.log(2))
    js.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(model.entropy(p)).all()


def test_primary_architecture_masking_freezing_gradients_and_checkpoint(tmp_path):
    net = detector().train()
    assert not net.wavlm.training and not net.clap.training
    assert net.adapter.training and net.classifier.training
    assert [layer.normalized_shape for layer in net.classifier if isinstance(layer, nn.LayerNorm)] == [(35,), (64,)]
    assert [(layer.in_features, layer.out_features) for layer in net.classifier if isinstance(layer, nn.Linear)] == [(35, 64), (64, 16), (16, 1)]
    assert [(layer.in_features, layer.out_features) for layer in net.adapter if isinstance(layer, nn.Linear)] == [(768, 512), (512, 512)]
    assert all(layer.p == .2 for component in (net.adapter, net.classifier)
               for layer in component if isinstance(layer, nn.Dropout))
    original = {name: value.clone() for name, value in net.state_dict().items() if name.startswith(("wavlm.", "clap.", "prototypes"))}
    inputs = batch()
    net.eval()
    output = net(inputs)
    with torch.no_grad():
        hidden = net.wavlm(**inputs["wavlm"], return_dict=True).last_hidden_state
    torch.testing.assert_close(output["z_w"][0], hidden[0, :2].mean(0))
    torch.testing.assert_close(output["p_w"], (output["v_w"] @ net.prototypes.T / .07).softmax(-1))
    torch.testing.assert_close(output["p_c"], (output["v_c"] @ net.prototypes.T / .07).softmax(-1))
    assert output["features"].shape == (2, 35) and output["logit"].shape == (2,)
    torch.testing.assert_close(net.spoof_score(output), output["logit"])
    altered = copy.deepcopy(inputs)
    altered["wavlm"]["input_values"][0, 2] = -900
    torch.testing.assert_close(net(altered)["logit"], output["logit"])
    altered["wavlm"]["attention_mask"][0] = 0
    with pytest.raises(ValueError, match="validity mask"):
        net(altered)
    net.train()
    optimizer = training.make_optimizer(net)
    optimizer.zero_grad()
    net.objective(net(inputs), inputs["labels"])["total"].backward()
    assert all(p.grad is None for encoder in (net.wavlm, net.clap) for p in encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for component in (net.adapter, net.classifier)
               for p in component.parameters())
    optimizer.step()
    for name, value in original.items():
        torch.testing.assert_close(net.state_dict()[name], value, rtol=0, atol=0)
    path = tmp_path / "synthetic.pt"
    torch.save(net.state_dict(), path)
    reloaded = detector()
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(net.eval()(inputs)["logit"], reloaded.eval()(inputs)["logit"], rtol=0, atol=0)


def test_losses_match_equations_and_all_spoof_batch_has_zero_prior():
    logits = torch.tensor([1., -2.], requires_grad=True)
    p = torch.softmax(torch.arange(16.).reshape(2, 8), -1).requires_grad_()
    q = torch.softmax(-torch.arange(16.).reshape(2, 8), -1)
    labels = torch.tensor([0, 1])
    output = {"logit": logits, "p_w": p, "p_c": q}
    losses = model.LACFLoss()(output, labels)
    det = F.binary_cross_entropy_with_logits(logits, labels.float())
    sem = F.binary_cross_entropy(p[:, 3:].sum(-1), labels.float())
    bf = model.js_divergence(p[:1], q[:1]).mean()
    torch.testing.assert_close(losses["total"], det + .25*sem + .1*bf)
    assert model.LACFLoss()(output, torch.ones(2))["bonafide"].item() == 0
    losses["total"].backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("groups,semantic,bf,expected", [
    (("p_c",), False, False, 8), (("p_w",), True, False, 8),
    (("p_w", "p_c"), True, False, 16),
    (("p_w", "p_c", "difference", "agreement"), True, False, 32),
    (tuple(model.FEATURE_DIMS), True, False, 35)])
def test_explicit_component_configuration_and_applicable_losses(groups, semantic, bf, expected):
    config = model.ComponentConfig(feature_groups=groups, semantic_loss=semantic, bonafide_loss=bf)
    net = detector(config)
    output = net(batch())
    assert output["features"].shape == (2, expected)
    assert net.classifier[1].in_features == expected
    assert (net.adapter is not None) == config.needs_wavlm
    assert (net.clap is not None) == config.needs_clap_audio
    losses = net.objective(output, batch()["labels"])
    assert torch.isfinite(losses["total"])
    if not semantic:
        assert losses["semantic"].item() == 0
    assert losses["bonafide"].item() == 0
    description = net.configuration()
    assert tuple(description["feature_groups"]) == groups
    assert description["fusion"]["output_dim"] == expected and description["stage"] == "frozen"


@pytest.mark.parametrize("kwargs", [{"feature_groups": ()}, {"feature_groups": ("p_w", "p_w")},
    {"feature_groups": ("invented",)}, {"temperature": 0}, {"temperature": float("nan")},
    {"semantic_weight": -1}])
def test_bad_component_configs_fail(kwargs):
    with pytest.raises(ValueError):
        model.ComponentConfig(**kwargs)


def test_replacement_fusion_declares_dimension_and_config_without_changing_loop():
    class FixtureFusion(nn.Module):
        output_dim = 8

        def configuration(self):
            return {"kind": "synthetic_fixture", "output_dim": 8}

        def forward(self, views):
            return views["p_c"]

    net = detector(fusion=FixtureFusion())
    assert net.classifier[1].in_features == 8
    assert net.configuration()["fusion"]["kind"] == "synthetic_fixture"
    losses, updates = training.train_epoch(net, [batch()], training.make_optimizer(net), "cpu")
    assert updates == 1 and np.isfinite(losses["total"])


def test_common_crop_once_then_dual_resampling_with_original_short_signal(tmp_path, monkeypatch):
    path = tmp_path / "fixture.wav"
    waveform = np.arange(12 * 8000, dtype=np.float32) / 100000
    sf.write(path, np.stack((waveform, waveform), -1), 8000, subtype="FLOAT")
    received = []
    original = data.resample_audio

    def capture(common, source, target):
        received.append((common.clone(), source, target))
        return original(common, source, target)

    monkeypatch.setattr(data, "resample_audio", capture)
    w, c, label = data.LACFDataset([(str(path), 1)], training=False)[0]
    assert w.shape == (160000,) and c.shape == (480000,) and label == 1
    assert [item[2] for item in received] == [16000, 48000]
    torch.testing.assert_close(received[0][0], received[1][0], rtol=0, atol=0)
    torch.testing.assert_close(received[0][0], torch.from_numpy(waveform[8000:88000]))
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([high - 1]))
    received.clear()
    data.LACFDataset([(str(path), 0)], training=True)[0]
    torch.testing.assert_close(received[0][0], torch.from_numpy(waveform[16000:]))
    short = tmp_path / "short.wav"
    sf.write(short, waveform[:8000], 8000, subtype="FLOAT")
    w, c, _ = data.LACFDataset([(str(short), 0)], training=False)[0]
    assert len(w) == 16000 and len(c) == 48000


def test_domain_class_sampler_equal_bucket_mass_reproducible_keeps_last_batch():
    datasets = [data.LACFDataset([(f"{d}-{i}", label) for i, label in enumerate(labels)], training=True)
                for d, labels in enumerate(([0, 1, 1], [0, 0, 0, 1, 1]))]
    sampler = data.balanced_sampler(datasets, 1234)
    assert sampler.replacement and sampler.num_samples == 8
    assert list(sampler) == list(data.balanced_sampler(datasets, 1234))
    offset = 0
    for dataset in datasets:
        for label in (0, 1):
            indices = [offset+i for i, (_, y) in enumerate(dataset.records) if y == label]
            assert sampler.weights[indices].sum().item() == pytest.approx(1)
        offset += len(dataset)
    with pytest.raises(ValueError, match="both classes"):
        data.balanced_sampler([data.LACFDataset([("fixture", 0)], training=True)], 1234)
    # Paths are not decoded: inspect loader/sampler and collate synthetic lists separately.
    loader = data.make_loader(datasets, lambda x: x, training=True, seed=1234,
                             batch_size=3, num_workers=0)
    assert not loader.drop_last and len(loader) == 3 and loader.prefetch_factor is None


def test_checkpoint_processors_padding_and_mask_are_actual_local_implementations():
    w = Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True)
    c = SimpleNamespace(feature_extractor=ClapFeatureExtractor(
        truncation="rand_trunc", padding="repeatpad", frequency_min=50))
    collate = pretrained.DualViewCollator(w, c)
    out = collate([(torch.linspace(-.2, .2, 800), torch.ones(2400)*.1, 0),
                   (torch.ones(1600)*.3, torch.ones(4800)*.2, 1)])
    assert out["wavlm"]["input_values"].shape == (2, 1600)
    assert out["wavlm"]["attention_mask"].sum(1).tolist() == [800, 1600]
    torch.testing.assert_close(out["wavlm"]["input_values"][1], torch.ones(1600)*.3)
    assert out["clap"]["input_features"].dtype == torch.float32
    assert not out["clap"]["is_longer"].any()
    assert collate.configuration()["clap_padding"] == "repeatpad"
    json.dumps(collate.configuration())
    with pytest.raises(ValueError, match="too short"):
        collate([(torch.ones(399), torch.ones(1197), 0)])


def test_accumulation_divides_by_actual_microbatches_and_preserves_short_final_group():
    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.))

        def forward(self, inputs):
            return self.weight * inputs["value"]

        def objective(self, output, labels):
            loss = (output - labels).square().mean()
            return {key: loss for key in ("total", "detection", "semantic", "bonafide")}

    net = ScalarModel()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    loader = [{"value": torch.ones(1), "labels": torch.tensor([float(y)])} for y in (1, 3, 2)]
    losses, updates = training.train_epoch(net, loader, optimizer, "cpu", accumulation_steps=2)
    # First update: average targets 1/3 -> weight=.4; final single batch -> weight=.72.
    assert net.weight.item() == pytest.approx(.72) and updates == 2
    assert np.isfinite(losses["total"])


def test_optimizer_excludes_frozen_parameters_and_uses_prescribed_adamw():
    net = detector()
    optimizer = training.make_optimizer(net)
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (3e-4, 1e-4)
    assert {id(p) for p in group["params"]} == {id(p) for p in net.parameters() if p.requires_grad}


def test_actual_tiny_wavlm_converts_sample_masks_to_feature_frame_masks():
    # Random tiny encoder configuration: no pretrained weight load or GPU call.
    encoder = WavLMModel(WavLMConfig(hidden_size=768, num_hidden_layers=1,
        intermediate_size=16, num_attention_heads=12, conv_dim=(4,), conv_stride=(2,),
        conv_kernel=(3,), num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=8,
        feat_extract_norm="layer", apply_spec_augment=False))
    net = model.LACF(encoder, FakeClap(), torch.randn(8, 512)).eval()
    inputs = batch()
    inputs["wavlm"] = {"input_values": torch.randn(2, 64),
                       "attention_mask": torch.arange(64)[None, :] < torch.tensor([[40], [64]])}
    with torch.no_grad():
        hidden = encoder(**inputs["wavlm"], return_dict=True).last_hidden_state
        frame_mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], inputs["wavlm"]["attention_mask"])
        expected = torch.stack([hidden[i, frame_mask[i]].mean(0) for i in range(2)])
        output = net(inputs)
    assert frame_mask.shape != inputs["wavlm"]["attention_mask"].shape
    assert frame_mask.sum(1).tolist() == [19, 31]
    torch.testing.assert_close(output["z_w"], expected)


def test_local_checkpoint_loading_is_offline_and_exact_prompt_order(monkeypatch):
    calls = []
    wavlm, clap = FakeWavLM(), FakeClap()
    wavlm.config.num_hidden_layers, wavlm.config._commit_hash = 12, "fixture-w"
    clap.config.audio_config, clap.config._commit_hash = SimpleNamespace(enable_fusion=False), "fixture-c"
    wavlm.config.to_dict = lambda: {"hidden_size": 768, "num_hidden_layers": 12}
    clap.config.to_dict = lambda: {"projection_dim": 512, "enable_fusion": False}
    w_processor = Wav2Vec2FeatureExtractor(do_normalize=False)
    prompts = []

    class FixtureProcessor:
        feature_extractor = ClapFeatureExtractor(truncation="rand_trunc", frequency_min=50)

        def __call__(self, *, text, padding, return_tensors):
            prompts.extend(text)
            return {"input_ids": torch.ones(24, 3, dtype=torch.long)}

    def loader(name, **kwargs):
        calls.append((name, kwargs))
        return wavlm if name == model.WAVLM_NAME else clap

    monkeypatch.setattr(pretrained.WavLMModel, "from_pretrained", loader)
    monkeypatch.setattr(pretrained.ClapModel, "from_pretrained", loader)
    monkeypatch.setattr(pretrained.Wav2Vec2FeatureExtractor, "from_pretrained",
                        lambda name, **kwargs: (calls.append((name, kwargs)), w_processor)[1])
    monkeypatch.setattr(pretrained.ClapProcessor, "from_pretrained",
                        lambda name, **kwargs: (calls.append((name, kwargs)), FixtureProcessor())[1])
    clap.get_text_features = lambda **kwargs: torch.arange(1., 24*512+1).reshape(24, 512)
    net, _, revisions = pretrained.load_local_model()
    assert all(kwargs["local_files_only"] for _, kwargs in calls)
    assert calls[2][1]["revision"] == "fixture-w" and calls[3][1]["revision"] == "fixture-c"
    assert prompts == [t.format(concept=c) for c in model.CONCEPTS for t in model.TEMPLATES]
    assert net.prototypes.shape == (8, 512) and not net.prototypes.requires_grad
    assert revisions["wavlm_revision"] == "fixture-w" and revisions["clap_revision"] == "fixture-c"
    assert revisions["encoder_configs"]["wavlm"]["hidden_size"] == 768
    assert all(kwargs["dtype"] == torch.float32 for _, kwargs in calls[:2])


def test_training_guards_preserve_existing_work_and_require_verified_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1"])
    with pytest.raises(ValueError, match="Verified transfer"):
        train.main()
    assert not (tmp_path / "outputs").exists()
    directory = tmp_path / "outputs/lacf/f1/1234"
    directory.mkdir(parents=True)
    (directory / "unrelated.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        train.main()
    assert (directory / "unrelated.txt").read_text() == "preserve"


def test_training_cuda_guard_precedes_local_weights_or_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f3", "--confirm-cache-verified"])
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "load_local_model", lambda: pytest.fail("must not load weights"))
    monkeypatch.setattr(train, "read_dataset", lambda *args: pytest.fail("must not open data"))
    with pytest.raises(RuntimeError, match="CUDA"):
        train.main()
    assert not (tmp_path / "outputs").exists()


def test_primary_trainer_source_isolation_ties_stopping_completion_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f3", "--confirm-cache-verified"])
    monkeypatch.setattr(train, "MAX_EPOCHS", 4)
    monkeypatch.setattr(train, "PATIENCE", 1)
    original_device = torch.device
    monkeypatch.setattr(train.torch, "device", lambda *args, **kwargs: original_device("cpu"))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(train.torch.cuda, "get_device_name", lambda device: "synthetic CPU fixture")
    monkeypatch.setattr(train, "code_provenance", lambda root: {"source_sha256": {"fixture": "fixture"}})
    net = detector()
    collator = SimpleNamespace(configuration=lambda: {"fixture": True})
    monkeypatch.setattr(train, "load_local_model", lambda: (net, collator, {"fixture_revision": "local"}))
    opened = []

    def reader(domain, split, root):
        opened.append((domain, split))
        assert domain in FOLDS["f3"]["sources"] and split in ("train", "dev")
        return [(f"{domain}-{i}", i % 2) for i in range({"asv2019": 2, "cfad": 4, "speechfake": 8}[domain])]

    monkeypatch.setattr(train, "read_dataset", reader)
    monkeypatch.setattr(train, "make_loader", lambda *args, **kwargs: [batch()])
    trained = []

    def epoch(*args, **kwargs):
        assert kwargs["accumulation_steps"] == 8
        trained.append(True)
        return {"total": .1, "detection": .1, "semantic": 0., "bonafide": 0.}, 1

    monkeypatch.setattr(train, "train_epoch", epoch)
    # Equal macro on epoch 2: keep epoch 1, then validate that selected checkpoint again.
    eers = iter([.1, .2, .3] * 3)
    monkeypatch.setattr(train, "source_eer", lambda *args: next(eers))
    train.main()
    assert opened == [(domain, split) for domain in FOLDS["f3"]["sources"] for split in ("train", "dev")]
    assert len(trained) == 2
    directory = tmp_path / "outputs/lacf/f3/1234"
    completion, digest = verify_completed_run(directory, "f3", 1234)
    assert completion["best_epoch"] == 1 and completion["final_epoch"] == 2
    assert completion["best_macro_source_dev_eer"] == pytest.approx(.2)
    assert completion["best_checkpoint_sha256"] == digest
    config = json.loads((directory / "config.json").read_text())
    assert config["selected_epoch"] == 1 and config["selected_checkpoint_sha256"] == digest
    assert config["final_seed_plan"] == [1234, 2345, 3456]
    assert config["components"]["fusion"]["output_dim"] == 35
    assert config["precision"] == "float32" and config["effective_batch_size"] == 32
    assert [config["dataset_counts"][domain]["dev"] for domain in FOLDS["f3"]["sources"]] == [2, 4, 8]
    assert not list(directory.glob("*scores.csv")) and not (directory / "metrics.json").exists()
    assert (directory / "training.log").is_file()


@pytest.mark.parametrize("arguments", [["--seed", "99"], ["--num-workers", "-1"], ["--prefetch-factor", "0"]])
def test_trainer_rejects_undeclared_seeds_or_invalid_loaders(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1", *arguments])
    with pytest.raises(SystemExit):
        train.parse_args()
