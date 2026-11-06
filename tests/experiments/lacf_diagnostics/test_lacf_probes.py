"""Synthetic CPU tests for the source-only Step 3 probe study."""

import hashlib
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.probes import (
    HEADS, INPUTS, SEEDS, FeatureBatches, FeatureSplit, Probe, inner_domains, load_split, score_probe,
)
from audio_deepfake_detection.metrics import compute_auroc, compute_eer
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.lacf import train_probes


def features(count=8, *, training=True, seed=1):
    generator = torch.Generator().manual_seed(seed)
    views = 3 if training else 1
    labels = torch.arange(count) % 2
    # Deliberately unsorted IDs exercise shared CSV canonical ordering.
    ids = [f"{count - i:04d}.wav" for i in range(count)]
    return FeatureSplit([(f"/{value}", int(label)) for value, label in zip(ids, labels)], ids, labels,
                        torch.arange(count * views).reshape(count, views),
                        torch.randn(count * views, 13, 768, generator=generator),
                        torch.randn(count * views, 512, generator=generator))


@pytest.mark.parametrize("inputs", INPUTS)
@pytest.mark.parametrize("head", HEADS)
def test_probes_have_only_fresh_head_and_optional_layer_weights(inputs, head):
    torch.manual_seed(1234)
    model = Probe(inputs, head)
    data = features()
    loader = FeatureBatches([data], inputs, "cpu", training=True, seed=1234)
    batch = next(iter(loader))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        losses = model.objective(output, batch["labels"])
    assert output["logit"].shape == (len(data.labels),)
    assert losses["total"].dtype == torch.float32
    assert losses["semantic"].item() == 0 and losses["bonafide"].item() == 0
    losses["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not data.wavlm.requires_grad and not data.clap.requires_grad
    assert not hasattr(model, "wavlm") and not hasattr(model, "clap") and not hasattr(model, "adapter")
    expected_dim = 512 if inputs == "clap" else 1280 if inputs == "concat" else 768
    count = expected_dim + 1 if head == "logistic" else 66 * expected_dim + 1249
    if inputs in ("weighted", "concat"):
        count += 13
        assert model.layer_weights.grad.abs().sum() > 0
        assert model.layer_weights.detach().softmax(0).tolist() == pytest.approx([1 / 13] * 13)
    assert model.configuration()["trainable_parameters"] == count


def test_raw_wavlm_and_unit_clap_policies_and_linear_weighted_mean():
    w = Probe("last", "logistic")
    with torch.no_grad():
        w.classifier.weight.fill_(1 / 768)
        w.classifier.bias.zero_()
    one = w({"wavlm": torch.ones(2, 768)})["logit"]
    two = w({"wavlm": 2 * torch.ones(2, 768)})["logit"]
    torch.testing.assert_close(two, 2 * one)
    c = Probe("clap", "logistic")
    raw = torch.randn(2, 512)
    torch.testing.assert_close(c({"clap": raw})["logit"], c({"clap": 10 * raw})["logit"])
    mixed = Probe("weighted", "logistic")
    states = torch.randn(3, 13, 7, 768)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1] * 7])
    pooled = (states * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
    before_pool = (states.mean(1) * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    torch.testing.assert_close(mixed({"wavlm": pooled})["logit"], mixed.classifier(before_pool).squeeze(-1))


def test_inner_splits_exclude_outer_domain_and_all_validation_training_overlap():
    for fold in ("f1", "f2", "f3"):
        for held in FOLDS[fold]["sources"]:
            fit = inner_domains(fold, held)
            assert len(fit) == 2 and held not in fit and FOLDS[fold]["target"] not in (*fit, held)
        with pytest.raises(ValueError):
            inner_domains(fold, FOLDS[fold]["target"])
    with pytest.raises(ValueError):
        inner_domains("f4", "asv5")
    assert SEEDS == (1234, 1235, 1236)


def test_sampling_mass_counts_utterances_not_fixed_views_and_dev_is_centered():
    a, b = features(12), features(20, seed=2)
    a.records = [(p, 0 if i < 2 else 1) for i, (p, _) in enumerate(a.records)]
    a.labels = torch.tensor([label for _, label in a.records])
    batcher = FeatureBatches([a, b], "concat", "cpu", training=True, seed=1234, batch_size=7)
    assert batcher.count == 32 and batcher.sampler.num_samples == 32 and len(batcher) == 5
    weights = batcher.sampler.weights
    for offset, d in ((0, a), (12, b)):
        for label in (0, 1):
            assert weights[offset:offset + len(d.labels)][d.labels == label].sum().item() == pytest.approx(1)
    assert sum(len(batch["labels"]) for batch in batcher) == 32
    short = features(8)
    short.view_rows[:] = short.view_rows[:, :1].expand(-1, 3)
    batch = next(iter(FeatureBatches([short], "last", "cpu", training=True, seed=1234)))
    assert all(any(torch.equal(row, short.wavlm[i, -1]) for i in short.view_rows[:, 0]) for row in batch["wavlm"])
    dev = features(9, training=False)
    batches = [dict((k, v.clone()) for k, v in value.items()) for value in
               FeatureBatches([dev], "concat", "cpu", training=False, seed=1234, batch_size=4)]
    torch.testing.assert_close(torch.cat([v["wavlm"] for v in batches]), dev.wavlm)
    torch.testing.assert_close(torch.cat([v["labels"] for v in batches]), dev.labels)


def synthetic_cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    directory = root / "asv2019/train"
    chunk = directory / "chunk_00000"
    chunk.mkdir(parents=True)
    data_root = tmp_path / "audio"
    canonical = [(str(data_root / "asvspoof2019/extracted/b.wav"), 0),
                 (str(data_root / "asvspoof2019/extracted/a.wav"), 1)]
    ids = ["b.wav", "a.wav"]
    metadata = [{"id": "asv2019/train/b.wav", "label": 0, "index": 0, "view_rows": [0, 0, 0]},
                {"id": "asv2019/train/a.wav", "label": 1, "index": 1, "view_rows": [1, 2, 3]}]
    (chunk / "records.json").write_text(json.dumps(metadata))
    w = np.arange(4 * 13 * 768, dtype=np.float32).reshape(4, 13, 768)
    c = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
    np.save(chunk / "wavlm_layers.npy", w)
    np.save(chunk / "clap_projected.npy", c)
    marker = {"records": 2, "unique_rows": 4,
              "files_sha256": {p.name: checkpoint_digest(p) for p in chunk.iterdir()}}
    (chunk / "complete.json").write_text(json.dumps(marker))
    manifest = {"records": 2, "unique_rows": 4,
                "canonical_ids_labels_sha256": hashlib.sha256(json.dumps(list(zip(ids, [0, 1]))).encode()).hexdigest(),
                "chunks": [{"path": chunk.name, "records": 2, "unique_rows": 4,
                            "completion_sha256": checkpoint_digest(chunk / "complete.json")}]}
    (directory / "complete.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("audio_deepfake_detection.lacf.probes.read_dataset", lambda d, s, r: canonical)
    return root, manifest, data_root, chunk, w, c


def test_cache_loading_validates_payloads_membership_and_crop_aliases(tmp_path, monkeypatch):
    root, manifest, data_root, chunk, w, c = synthetic_cache(tmp_path, monkeypatch)
    data = load_split(root, "asv2019", "train", manifest, data_root)
    np.testing.assert_array_equal(data.wavlm.numpy(), w)
    np.testing.assert_array_equal(data.clap.numpy(), c)
    assert data.view_rows.tolist() == [[0, 0, 0], [1, 2, 3]] and data.ids == ["b.wav", "a.wav"]
    with (chunk / "clap_projected.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_split(root, "asv2019", "train", manifest, data_root)
    with pytest.raises(ValueError, match="restricted"):
        load_split(root, "asv2019", "eval", manifest, data_root)


def test_compact_last_layer_loading_and_batches_match_full_cache_exactly(tmp_path, monkeypatch):
    root, manifest, data_root, _, _, _ = synthetic_cache(tmp_path, monkeypatch)
    full = load_split(root, "asv2019", "train", manifest, data_root)
    compact = load_split(root, "asv2019", "train", manifest, data_root, last_layer_only=True)
    assert compact.wavlm.shape == (4, 768)
    torch.testing.assert_close(compact.wavlm, full.wavlm[:, -1], rtol=0, atol=0)
    for training in (False, True):
        options = dict(training=training, seed=1234, last_layer_only=True)
        left = next(iter(FeatureBatches([full], "concat", "cpu", **options)))
        right = next(iter(FeatureBatches([compact], "concat", "cpu", **options)))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="thirteen"):
        FeatureBatches([compact], "weighted", "cpu", training=True, seed=1234)


@pytest.mark.parametrize("fixed_scores", [False, True])
def test_source_fit_reload_export_strict_ties_and_no_overwrite(tmp_path, monkeypatch, fixed_scores):
    bank = {("asv5", "train"): features(4), ("cfad", "train"): features(4, seed=2),
            ("asv2019", "dev"): features(4, training=False, seed=3)}
    if fixed_scores:
        def score(_model, loader):
            values = torch.tensor([.4, -.3, .2, .8])
            labels = loader.datasets[0].labels.numpy()
            return values, {"eer": compute_eer(labels, values.numpy()), "auroc": compute_auroc(labels, values.numpy())}
        monkeypatch.setattr(train_probes, "score_probe", score)
        monkeypatch.setattr(train_probes, "MAX_EPOCHS", 8)
    else:
        monkeypatch.setattr(train_probes, "MAX_EPOCHS", 2)
    run = tmp_path / "last_mlp/inner_asv2019/f1/1234"
    logger = logging.getLogger(f"test_probe_{fixed_scores}")
    logger.setLevel(logging.INFO)
    options = dict(fold="f1", held="asv2019", inputs="last", head="mlp", seed=1234,
                   provenance={"test": "synthetic CPU only"}, device="cpu", logger=logger)
    result = train_probes.fit_probe(bank, run, **options)
    if fixed_scores:
        assert result["best_epoch"] == 1 and result["final_epoch"] == 6
    else:
        assert result["final_epoch"] == 2
    train_probes.verify_probe_completion(run, "f1", 1234)
    original = checkpoint_digest(run / "best.pt")
    with pytest.raises(FileExistsError):
        train_probes.fit_probe(bank, run, **options)
    assert checkpoint_digest(run / "best.pt") == original
    with (run / "best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="verification"):
        train_probes.verify_probe_completion(run, "f1", 1234)


def test_summary_averages_rotations_before_seed_sample_std():
    results = []
    for seed_index, seed in enumerate(SEEDS):
        for inner_index, domain in enumerate(FOLDS["f1"]["sources"]):
            eer = (1 + seed_index + inner_index) / 10
            results.append({"fold": "f1", "inputs": "last", "head": "logistic", "seed": seed,
                            "inner_held_domain": domain, "eer": eer, "auroc": 1 - eer})
    summary = train_probes.aggregate(results)
    row = summary["three_seed_summary"][0]
    assert row["eer_mean"] == pytest.approx(.3) and row["eer_std"] == pytest.approx(.1)
    assert row["auroc_mean"] == pytest.approx(.7) and row["auroc_std"] == pytest.approx(.1)
    assert train_probes.aggregate(results[:-1])["three_seed_summary"] == []
