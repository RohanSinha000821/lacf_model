"""Synthetic CPU tests: no real datasets, pretrained downloads or CUDA work."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_deepfake_detection import protocol
from audio_deepfake_detection.sota import ssl_gating as ssl
from scripts.ssl_gating import evaluate, train


class Frontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.linspace(0.5, 1.5, 1024))
        self.config = SimpleNamespace(hidden_size=1024, num_hidden_layers=24,
                                      to_dict=lambda: {"hidden_size": 1024, "num_hidden_layers": 24})

    def forward(self, waves, *, output_hidden_states):
        assert output_hidden_states
        # Depend on the last sample as well as the first; retain varied time steps.
        frames = waves[:, ::100].unsqueeze(-1) * self.scale
        return SimpleNamespace(hidden_states=tuple(frames + i * 0.01 for i in range(25)))


def test_cka_matches_centered_gram_reference_and_keeps_gradients():
    torch.manual_seed(7)
    a, b = torch.randn(5, 8, requires_grad=True), torch.randn(5, 12, requires_grad=True)
    h = torch.eye(5) - 0.2
    k, l = a @ a.T, b @ b.T
    expected = torch.trace(k @ h @ l @ h) / (
        (torch.trace(k @ h @ k @ h) * torch.trace(l @ h @ l @ h)).sqrt() + 1e-6)
    observed = ssl.linear_cka(a, b)
    torch.testing.assert_close(observed, expected)
    observed.backward()
    assert a.grad.norm() > 0 and b.grad.norm() > 0


@pytest.mark.parametrize("batch", [1, 5])
def test_degenerate_cka_has_finite_zero_gradients(batch):
    a, b = torch.ones(batch, 128, requires_grad=True), torch.ones(batch, 128, requires_grad=True)
    loss = ssl.linear_cka(a, b)
    loss.backward()
    assert loss == 0
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_pairwise_cka_has_all_six_pairs_and_backpropagates():
    means = torch.randn(5, 4, 128, requires_grad=True)
    expected = sum(ssl.linear_cka(means[:, i], means[:, j]).abs()
                   for i in range(4) for j in range(i + 1, 4)) / 6
    loss = ssl.cka_loss(means)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(means.grad[:, i].norm() > 0 for i in range(4))


def test_model_updates_frontend_and_all_blocks_and_reloads(tmp_path):
    torch.manual_seed(1234)
    model = ssl.SSLFeatureGating(Frontend())
    waves, labels = torch.randn(5, 800), torch.tensor([0, 1, 1, 1, 0])
    logits, means = model(waves, return_block_means=True)
    assert logits.shape == (5, 2) and means.shape == (5, 4, 128)
    optimizer = train.make_optimizer(model)
    cka = ssl.cka_loss(means)
    # The repaired auxiliary term reaches the actual backbone, independently of CE.
    gradient = torch.autograd.grad(cka, model.feature_extractor.scale, retain_graph=True)[0]
    assert gradient.norm() > 0
    loss = torch.nn.functional.cross_entropy(logits, labels, weight=torch.tensor([0.9, 0.1])) + cka
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(block.channel_proj1[0].weight.grad.norm() > 0 for block in model.blocks)
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == 3e-6
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    model.eval()
    with torch.inference_mode():
        expected = model(waves)
    path = tmp_path / "synthetic.pt"
    torch.save(model.state_dict(), path)
    restored = ssl.SSLFeatureGating(Frontend()).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(waves), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.spoof_score(expected), expected[:, 1] - expected[:, 0])


def test_block_concatenation_keeps_time_aligned_and_labels_reordered():
    model = ssl.SSLFeatureGating(Frontend()).eval()
    seen, block_outputs = [], []
    hooks = [block.register_forward_hook(lambda _, args, out: block_outputs.append(out))
             for block in model.blocks]
    hooks.append(model.pooling.register_forward_pre_hook(lambda _, args: seen.append(args[0])))
    raw_logits = []
    hooks.append(model.classifier.register_forward_hook(lambda _, args, out: raw_logits.append(out)))
    with torch.inference_mode():
        actual = model(torch.randn(2, 800))
    assert len(block_outputs) == 4
    for i in range(4):
        torch.testing.assert_close(seen[0][:, :, i * 128:(i + 1) * 128], block_outputs[i], rtol=0, atol=0)
    torch.testing.assert_close(actual, raw_logits[0].flip(1), rtol=0, atol=0)
    for hook in hooks:
        hook.remove()


def test_full_audio_dataset_resamples_and_does_not_crop_or_normalize(tmp_path):
    path = tmp_path / "full.wav"
    wave = np.full((35001, 2), 0.2, dtype=np.float32)
    wave[-100:] = 0.7
    sf.write(path, wave, 8000, subtype="FLOAT")
    output, label = ssl.FullAudioDataset([(str(path), 1)])[0]
    assert len(output) == 70002 and label == 1
    assert output[100:1000].mean() == pytest.approx(0.2, abs=1e-4)
    assert output[-100:-10].mean() > 0.65


def test_dynamic_zero_padding_and_batch_one_do_not_repeat_short_audio():
    waves, labels = ssl.collate_audio([(torch.tensor([1.0, 2.0]), 0), (torch.arange(5.0), 1)])
    torch.testing.assert_close(waves[0], torch.tensor([1., 2., 0., 0., 0.]))
    assert labels.tolist() == [0, 1]
    short = [(torch.arange(3.0), 0)]
    loader = ssl.make_eval_loader(short, num_workers=0)
    assert next(iter(loader))[0].shape == (1, 3)


def test_equal_domain_natural_class_sampler_and_no_dropped_batch():
    sets = [[(torch.arange(10.0 + i), j % 2) for j in range(n)] for i, n in enumerate((2, 4, 6))]
    loader = ssl.make_train_loader(sets, num_workers=0, seed=1234)
    other = ssl.make_train_loader(sets, num_workers=0, seed=1234)
    assert loader.batch_size == 5 and loader.sampler.replacement
    assert loader.sampler.num_samples == 12
    assert [loader.sampler.weights[a:b].sum().item() for a, b in [(0, 2), (2, 6), (6, 12)]] == pytest.approx([1., 1., 1.])
    assert list(loader.sampler) == list(other.sampler)
    assert [len(labels) for _, labels in loader] == [5, 5, 2]
    assert loader.prefetch_factor is None and not loader.persistent_workers
    assert protocol.final_seeds_for("ssl_gating") == (1234, 2345, 3456)


def test_pretrained_loading_is_offline_pinned_and_never_detector_weights(monkeypatch):
    import transformers
    captured = []

    def load(name, **kwargs):
        captured.append((name, kwargs))
        return Frontend()

    monkeypatch.setattr(transformers.Wav2Vec2Model, "from_pretrained", load)
    ssl.SSLFeatureGating()
    assert captured == [("facebook/wav2vec2-xls-r-300m", {
        "revision": ssl.PRETRAINED_REVISION, "local_files_only": True, "attn_implementation": "sdpa"})]


def test_reconstruction_uses_checkpoint_config_without_loading_pretrained(monkeypatch):
    import transformers
    captured = []

    def construct(config):
        captured.append(config)
        return Frontend()

    monkeypatch.setattr(transformers, "Wav2Vec2Model", construct)
    ssl.SSLFeatureGating.from_frontend_config({"hidden_size": 1024, "num_hidden_layers": 24})
    assert captured[0]._attn_implementation == "sdpa"


@pytest.mark.parametrize("tokens", [["--seed", "-1"], ["--num-workers", "-1"], ["--prefetch-factor", "0"],
                                   ["--gpu-memory-budget-gib", "0"], ["--gpu-memory-budget-gib", "nan"]])
def test_bad_training_args(monkeypatch, tokens):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1", *tokens])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_nonfinite_loss_stops_before_update():
    model = ssl.SSLFeatureGating(Frontend())
    optimizer = train.make_optimizer(model)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        train.train_epoch(model, [(torch.full((5, 800), float("nan")), torch.tensor([0, 1, 0, 1, 1]))],
                          optimizer, torch.device("cpu"))


def test_trainer_selects_macro_source_eer_and_preserves_ties(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f3", "--num-workers", "0"])
    monkeypatch.setattr(train, "MAX_EPOCHS", 3)
    monkeypatch.setattr(train, "PATIENCE", 1)
    monkeypatch.setattr(train, "set_seed", lambda seed: None)
    monkeypatch.setattr(train, "code_provenance", lambda root: {})
    original_device = torch.device
    monkeypatch.setattr(train.torch, "device", lambda *args, **kw: original_device("cpu"))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(train.torch.cuda, "get_device_name", lambda device: "synthetic CPU")
    monkeypatch.setattr(train.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(train.torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(train.torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(train, "SSLFeatureGating", lambda: ssl.SSLFeatureGating(Frontend()))
    batch = [(torch.randn(5, 800), torch.tensor([0, 1, 1, 1, 0]))]
    monkeypatch.setattr(train, "make_train_loader", lambda *args, **kw: batch)
    monkeypatch.setattr(train, "make_eval_loader", lambda *args, **kw: batch)
    values = iter([0.0, 0.3, 0.6, 0.0, 0.3, 0.6])
    monkeypatch.setattr(train, "evaluate_eer", lambda *args: next(values))
    opened = []

    def read(domain, split, root):
        opened.append((domain, split))
        return [("synthetic.wav", 0)]

    monkeypatch.setattr(train, "read_dataset", read)
    train.main()
    directory = tmp_path / "outputs/ssl_gating/f3/1234"
    best = torch.load(directory / "best.pt", weights_only=True)
    last = torch.load(directory / "last.pt", weights_only=True)
    complete = json.loads((directory / "training_complete.json").read_text())
    assert best["epoch"] == 1 and last["epoch"] == 2 and last["bad_epochs"] == 1
    assert best["macro_dev_eer"] == pytest.approx(0.3)
    assert complete["best_checkpoint_sha256"] == protocol.checkpoint_digest(directory / "best.pt")
    assert best["config"]["rawboost"] is False and best["config"]["scheduler"] == "constant"
    assert best["config"]["trainable_parameter_count"] == best["config"]["parameter_count"]
    assert opened == [(domain, split) for domain in protocol.FOLDS["f3"]["sources"] for split in ("train", "dev")]
    assert not (directory / "target_scores.csv").exists()
    with pytest.raises(FileExistsError):
        train.main()


class Detector(torch.nn.Module):
    def forward(self, waveforms):
        return torch.stack((torch.zeros_like(waveforms[:, 0]), waveforms[:, 0]), dim=1)

    spoof_score = staticmethod(ssl.SSLFeatureGating.spoof_score)


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    directory = tmp_path / "outputs/ssl_gating/f1/1234"
    directory.mkdir(parents=True)
    config = {"run_name": "ssl_gating", "fold": "f1", "seed": 1234, "model_name": "SSLFeatureGating",
              "model_config": ssl.MODEL_CONFIG, "upstream_commit": ssl.UPSTREAM_COMMIT, "precision": "float32",
              "rawboost": False, "selection": "strictly_lower_macro_source_dev_eer",
              "frontend_config": {"hidden_size": 1024, "num_hidden_layers": 24},
              "input_policy": ssl.MODEL_CONFIG["input_policy"],
              "sources": list(protocol.FOLDS["f1"]["sources"]), "target": "speechfake", "target_split": "test"}
    checkpoint = {"config": config, "fold": "f1", "epoch": 2, "macro_dev_eer": 0.0,
                  "model_state_dict": Detector().state_dict()}
    torch.save(checkpoint, directory / "best.pt")
    completion = {"fold": "f1", "seed": 1234, "best_epoch": 2, "final_epoch": 5,
                  "best_macro_source_dev_eer": 0.0,
                  "best_checkpoint_sha256": protocol.checkpoint_digest(directory / "best.pt")}
    (directory / "training_complete.json").write_text(json.dumps(completion))
    opened = []

    def reader(domain, split, root):
        opened.append((domain, split))
        base = protocol.dataset_root(root, domain)
        return [(str(base / f"{split}/{i}.wav"), label) for i, label in enumerate((0, 0, 1, 1))]

    monkeypatch.setattr(evaluate, "read_dataset", reader)
    monkeypatch.setattr(protocol, "read_dataset", reader)
    monkeypatch.setattr(evaluate.SSLFeatureGating, "from_frontend_config", lambda config: Detector())
    monkeypatch.setattr(evaluate, "FullAudioDataset", lambda records: records)
    monkeypatch.setattr(evaluate, "make_eval_loader", lambda *args, **kw: [
        (torch.tensor([[value]]), torch.tensor([label]))
        for value, label in [(-2., 0), (-1., 0), (1., 1), (2., 1)]])
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "f1", "--bootstrap-resamples", "8"])
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: True)
    original_device = torch.device
    monkeypatch.setattr(evaluate.torch, "device", lambda *args, **kw: original_device("cpu"))
    return directory, opened, checkpoint, completion


def test_native_export_uses_common_metrics_after_source_calibration(evaluation_run):
    directory, opened, _, _ = evaluation_run
    evaluate.main()
    assert opened == [(d, "dev") for d in protocol.FOLDS["f1"]["sources"]] + [("speechfake", "test")]
    report = json.loads((directory / "metrics.json").read_text())
    assert report["target"]["eer"] == 0.0 and report["target"]["auroc"] == 1.0
    assert report["unlabeled_zscore_transfer"]["transductive"] is True
    assert report["confidence_intervals"]["source_thresholds_fixed_within_resamples"] is True
    assert report["input_policy"] == ssl.MODEL_CONFIG["input_policy"]
    manifest = json.loads((directory / "score_manifest.json").read_text())
    assert len(manifest["score_sha256"]) == 4
    with pytest.raises(FileExistsError):
        evaluate.main()


def test_source_reload_disagreement_blocks_target(evaluation_run):
    directory, opened, checkpoint, completion = evaluation_run
    checkpoint["macro_dev_eer"] = completion["best_macro_source_dev_eer"] = 0.2
    torch.save(checkpoint, directory / "best.pt")
    completion["best_checkpoint_sha256"] = protocol.checkpoint_digest(directory / "best.pt")
    (directory / "training_complete.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="Source-dev macro"):
        evaluate.main()
    assert opened == [(d, "dev") for d in protocol.FOLDS["f1"]["sources"]]


@pytest.mark.parametrize("key,value", [("seed", 2345), ("precision", "bfloat16"), ("rawboost", True),
                                       ("run_name", "other"), ("target_split", "dev")])
def test_checkpoint_identity_and_recipe_mismatch_rejected(evaluation_run, key, value):
    _, _, checkpoint, completion = evaluation_run
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["config"][key] = value
    with pytest.raises(ValueError):
        evaluate.verify_checkpoint(checkpoint, completion, "f1", 1234, "ssl_gating")


def test_registered_source_score_tampering_is_rejected(evaluation_run):
    directory, _, _, _ = evaluation_run
    evaluate.main()
    # Model an interrupted report-writing phase, retaining the registered CSVs.
    (directory / "metrics.json").unlink()
    path = directory / "source_dev_asv2019.csv"
    path.write_text(path.read_text().replace("-2.0", "-20.0"))
    with pytest.raises(ValueError, match="unregistered or changed"):
        evaluate.main()
