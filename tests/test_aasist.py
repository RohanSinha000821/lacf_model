"""Synthetic CPU checks only: never open real datasets or allocate CUDA memory."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_deepfake_detection import protocol
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from audio_deepfake_detection.sota import aasist
from audio_deepfake_detection.sota.aasist_arch import Model
from scripts.aasist import evaluate, train


def test_upstream_architecture_is_unmodified_and_licensed():
    root = Path(aasist.__file__).parent
    assert hashlib.sha256((root / "aasist_arch.py").read_bytes()).hexdigest() == (
        "9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50")
    assert "Copyright (c) 2021-present NAVER Corp." in (root / "AASIST_LICENSE").read_text()


def test_short_audio_is_repeated_not_zero_padded():
    waveform = torch.tensor([1.0, -1.0, 0.25])
    output = aasist.fixed_length_audio(waveform, training=False)
    assert output.shape == (64600,)
    torch.testing.assert_close(output[:6], waveform.repeat(2))


def test_eval_uses_first_segment_and_training_accepts_exact_length(monkeypatch):
    waveform = torch.arange(64602, dtype=torch.float32)
    torch.testing.assert_close(aasist.fixed_length_audio(waveform, training=False), waveform[:64600])
    requested = []

    def choose_last(high, size):
        requested.append(high)
        return torch.tensor([high - 1])

    monkeypatch.setattr(aasist.torch, "randint", choose_last)
    torch.testing.assert_close(aasist.fixed_length_audio(waveform, training=True), waveform[2:])
    torch.testing.assert_close(aasist.fixed_length_audio(waveform[:64600], training=True), waveform[:64600])
    assert requested == [3, 1]


@pytest.mark.parametrize("waveform", [torch.tensor([]), torch.tensor([float("nan")]), torch.zeros(1, 2)])
def test_bad_waveforms_rejected(waveform):
    with pytest.raises(ValueError):
        aasist.fixed_length_audio(waveform, training=True)


def test_dataset_resamples_stereo_and_preserves_labels(tmp_path):
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full((8000, 2), 0.25, dtype=np.float32), 8000, subtype="FLOAT")
    waveform, label = aasist.AASISTDataset([(str(path), 1)], training=False)[0]
    assert waveform.shape == (64600,) and waveform.dtype == torch.float32 and label == 1
    assert waveform[100:1000].mean().item() == pytest.approx(0.25, abs=1e-3)
    assert aasist.AASISTDataset([(str(path), 0)], training=True)[0][1] == 0


def test_domain_sampler_equal_weights_reproducible_and_no_dropped_draws():
    sets = [torch.utils.data.TensorDataset(torch.zeros(n, 64600), torch.zeros(n, dtype=torch.long))
            for n in (2, 4, 6)]
    loader = aasist.make_train_loader(sets, batch_size=5, seed=1234, num_workers=0)
    sampler = loader.sampler
    assert sampler.replacement and sampler.num_samples == 12
    assert [float(sampler.weights[:2].sum()), float(sampler.weights[2:6].sum()),
            float(sampler.weights[6:].sum())] == pytest.approx([1, 1, 1])
    other = aasist.make_train_loader(sets, batch_size=5, seed=1234, num_workers=0)
    assert list(sampler) == list(other.sampler)
    assert [len(labels) for _, labels in loader] == [5, 5, 2]
    assert loader.prefetch_factor is None and not loader.persistent_workers
    assert aasist.make_eval_loader(sets[0], num_workers=0).batch_size == 1


def test_model_matches_authors_logits_after_label_reordering_and_reloads(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(1234)
        model = aasist.AASIST().eval()
        waveform = torch.randn(1, 64600)
        with torch.inference_mode():
            _, authors_logits = Model.forward(model, waveform, Freq_aug=False)
            logits = model(waveform)
        torch.testing.assert_close(logits, authors_logits[:, [1, 0]], rtol=0, atol=0)
        torch.testing.assert_close(model.spoof_score(logits), authors_logits[:, 0] - authors_logits[:, 1])
        saved = tmp_path / "synthetic.pt"
        torch.save(model.state_dict(), saved)
        reloaded = aasist.AASIST().eval()
        reloaded.load_state_dict(torch.load(saved, weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(waveform), logits, rtol=0, atol=0)
        model.train()
        optimizer, scheduler = train.make_optimizer(model, steps_per_epoch=2)
        loss = torch.nn.functional.cross_entropy(model(waveform.repeat(2, 1)), torch.tensor([0, 1]),
                                                weight=torch.tensor([0.9, 0.1]))
        loss.backward()
        assert torch.isfinite(loss) and all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                           for parameter in model.parameters())
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] < 1e-4
    finally:
        torch.set_num_threads(previous_threads)


def test_cosine_schedule_reaches_floor_per_update():
    model = torch.nn.Linear(1, 2)
    optimizer, scheduler = train.make_optimizer(model, steps_per_epoch=1)
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


@pytest.mark.parametrize("tokens", [["--num-workers", "-1"], ["--seed", "-1"], ["--prefetch-factor", "0"]])
def test_invalid_training_args(monkeypatch, tokens):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1", *tokens])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_default_recipe_and_source_only_fold_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f3"])
    args = train.parse_args()
    assert args.run_name == "aasist" and args.seed == 1234
    assert (train.BATCH_SIZE, train.LEARNING_RATE, train.WEIGHT_DECAY, train.PATIENCE) == (24, 1e-4, 1e-4, 5)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train, "set_seed", lambda seed: None)
    monkeypatch.setattr(train, "make_eval_loader", lambda *args, **kwargs: None)
    opened = []

    def reader(domain, split, root):
        opened.append((domain, split))
        return [("synthetic.wav", 0)]

    def boundary(*args, **kwargs):
        assert kwargs["batch_size"] == 24
        raise RuntimeError("stop before model/GPU construction")

    monkeypatch.setattr(train, "read_dataset", reader)
    monkeypatch.setattr(train, "make_train_loader", boundary)
    with pytest.raises(RuntimeError, match="stop before"):
        train.main()
    assert opened == [(domain, split) for domain in FOLDS["f3"]["sources"] for split in ("train", "dev")]
    assert all(domain != "asv5" for domain, _ in opened)


def test_trainer_refuses_existing_checkpoint_before_opening_data(tmp_path, monkeypatch):
    directory = tmp_path / "outputs/aasist/f1/1234"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"preserve")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1"])
    with pytest.raises(FileExistsError):
        train.main()
    assert (directory / "best.pt").read_bytes() == b"preserve"


def test_trainer_writes_best_last_and_completion_with_source_only_early_stopping(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f2", "--num-workers", "0"])
    monkeypatch.setattr(train, "MAX_EPOCHS", 3)
    monkeypatch.setattr(train, "PATIENCE", 1)
    monkeypatch.setattr(train, "set_seed", lambda seed: None)
    monkeypatch.setattr(train, "code_provenance", lambda root: {})
    original_device = torch.device
    monkeypatch.setattr(train.torch, "device", lambda *args, **kwargs: original_device("cpu"))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    # Mocking the CUDA guard must not make Adam access a real GPU stream.
    monkeypatch.setattr(train.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(train.torch.cuda, "get_device_name", lambda device: "synthetic CPU test")
    monkeypatch.setattr(train.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(train.torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(train.torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(train, "AASIST", lambda: torch.nn.Linear(2, 2))
    batch = [(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([0, 1]))]
    monkeypatch.setattr(train, "make_train_loader", lambda *args, **kwargs: batch)
    monkeypatch.setattr(train, "make_eval_loader", lambda *args, **kwargs: batch)
    monkeypatch.setattr(train, "evaluate_eer", lambda *args: 0.1)
    opened = []

    def reader(domain, split, root):
        opened.append((domain, split))
        return [("synthetic.wav", 0)]

    monkeypatch.setattr(train, "read_dataset", reader)
    train.main()
    directory = tmp_path / "outputs/aasist/f2/1234"
    best = torch.load(directory / "best.pt", weights_only=True)
    last = torch.load(directory / "last.pt", weights_only=True)
    completion = json.loads((directory / "training_complete.json").read_text())
    assert best["epoch"] == 1 and last["epoch"] == 2 and last["bad_epochs"] == 1
    assert completion["best_epoch"] == 1 and completion["final_epoch"] == 2
    assert completion["best_checkpoint_sha256"] == checkpoint_digest(directory / "best.pt")
    assert last["optimizer_updates"] == 2 and last["scheduler_state_dict"]["last_epoch"] == 2
    assert best["config"]["precision"] == "float32"
    assert opened == [(domain, split) for domain in FOLDS["f2"]["sources"] for split in ("train", "dev")]
    assert not (directory / "target_scores.csv").exists()


class TinyDetector(torch.nn.Module):
    def forward(self, waveforms):
        return torch.stack((torch.zeros_like(waveforms[:, 0]), waveforms[:, 0]), dim=1)

    spoof_score = staticmethod(aasist.AASIST.spoof_score)


@pytest.fixture
def synthetic_evaluation(tmp_path, monkeypatch):
    directory = tmp_path / "outputs/aasist/f1/1234"
    directory.mkdir(parents=True)
    config = {"run_name": "aasist", "fold": "f1", "seed": 1234, "model_name": "AASIST",
              "model_config": aasist.MODEL_CONFIG, "upstream_commit": aasist.UPSTREAM_COMMIT,
              "precision": "float32", "eval_crop": "first_64600_repeat_if_short",
              "sources": list(FOLDS["f1"]["sources"]), "target": "speechfake", "target_split": "test"}
    checkpoint = {"config": config, "fold": "f1", "epoch": 2, "macro_dev_eer": 0.0,
                  "model_state_dict": TinyDetector().state_dict()}
    torch.save(checkpoint, directory / "best.pt")
    completion = {"fold": "f1", "seed": 1234, "best_epoch": 2, "final_epoch": 5,
                  "best_macro_source_dev_eer": 0.0, "best_checkpoint_sha256": checkpoint_digest(directory / "best.pt")}
    (directory / "training_complete.json").write_text(json.dumps(completion))
    opened = []

    def reader(domain, split, root):
        opened.append((domain, split))
        base = protocol.dataset_root(root, domain)
        return [(str(base / f"{split}/{index}.wav"), label) for index, label in enumerate((0, 0, 1, 1))]

    monkeypatch.setattr(evaluate, "read_dataset", reader)
    monkeypatch.setattr(protocol, "read_dataset", reader)
    monkeypatch.setattr(evaluate, "AASIST", TinyDetector)
    monkeypatch.setattr(evaluate, "AASISTDataset", lambda records, **kwargs: records)
    monkeypatch.setattr(evaluate, "make_eval_loader", lambda *args, **kwargs: [
        (torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), torch.tensor([0, 0, 1, 1]))])
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "f1", "--confirm-protocol-frozen",
                                     "--bootstrap-resamples", "8"])
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: True)
    original_device = torch.device
    monkeypatch.setattr(evaluate.torch, "device", lambda *args, **kwargs: original_device("cpu"))
    return directory, opened, checkpoint, completion


def test_score_export_to_common_report_and_reuse_checks(synthetic_evaluation):
    directory, opened, _, _ = synthetic_evaluation
    evaluate.main()
    assert opened == [(domain, "dev") for domain in FOLDS["f1"]["sources"]] + [("speechfake", "test")]
    report = json.loads((directory / "metrics.json").read_text())
    assert report["model"] == "aasist" and report["target"]["eer"] == 0.0
    assert report["unlabeled_zscore_transfer"]["transductive"] is True
    assert report["confidence_intervals"]["source_thresholds_fixed_within_resamples"] is True
    manifest = json.loads((directory / "score_manifest.json").read_text())
    assert len(manifest["score_sha256"]) == 4
    with pytest.raises(FileExistsError):
        evaluate.main()


def test_source_mismatch_blocks_target_opening(synthetic_evaluation):
    directory, opened, checkpoint, completion = synthetic_evaluation
    checkpoint["macro_dev_eer"] = completion["best_macro_source_dev_eer"] = 0.2
    torch.save(checkpoint, directory / "best.pt")
    completion["best_checkpoint_sha256"] = checkpoint_digest(directory / "best.pt")
    (directory / "training_complete.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="Source-dev macro"):
        evaluate.main()
    assert opened == [(domain, "dev") for domain in FOLDS["f1"]["sources"]]
    assert not (directory / "target_scores.csv").exists()


@pytest.mark.parametrize("key,value", [("seed", 2345), ("precision", "bfloat16"),
                                       ("run_name", "other"), ("target_split", "dev")])
def test_wrong_checkpoint_rejected(synthetic_evaluation, key, value):
    _, _, checkpoint, completion = synthetic_evaluation
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["config"][key] = value
    with pytest.raises(ValueError):
        evaluate.verify_checkpoint(checkpoint, completion, "f1", 1234, "aasist")


def test_target_evaluation_requires_explicit_freeze_confirmation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "f1"])
    with pytest.raises(SystemExit):
        evaluate.parse_args()
