"""Synthetic CPU checks for source-only fixed-crop diagnostic features."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from transformers import ClapFeatureExtractor

from audio_deepfake_detection.data import load_audio_segment, resample_audio
from audio_deepfake_detection.lacf.diagnostic_cache import (
    FixedCropDataset, all_layer_features, crop_intervals, grouped_views,
)
from scripts.lacf.cache_diagnostics import (
    COUNTS, encode_chunk, preserved_chunks, source_records, verify_scientific_identity,
)
from audio_deepfake_detection.protocol import checkpoint_digest


def test_crops_are_fixed_shared_across_seeds_and_do_not_change_global_rng():
    state = torch.random.get_rng_state().clone()
    intervals, aliases = crop_intervals(500000, 16000, "asv5/train/test", training=True)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.manual_seed(1236)
    assert crop_intervals(500000, 16000, "asv5/train/test", training=True) == (intervals, aliases)
    assert len(aliases) == 3 and intervals[aliases[0]] == (170000, 160000)
    assert all(0 <= start <= 340000 and length == 160000 for start, length in intervals)
    assert crop_intervals(500000, 16000, "asv5/train/test", training=False) == ([(170000, 160000)], [0])
    assert crop_intervals(24000, 16000, "short", training=True) == ([(0, 24000)], [0, 0, 0])
    with pytest.raises(ValueError):
        crop_intervals(0, 16000, "empty", training=True)


@pytest.mark.parametrize("frames", [12000, 91000])
def test_fixed_audio_matches_canonical_center_and_retains_short_views(tmp_path, frames):
    rate = 8000
    audio = np.random.default_rng(5).normal(0, .1, (frames, 2)).astype(np.float32)
    path = tmp_path / "audio.wav"
    sf.write(path, audio, rate, subtype="FLOAT")
    item = FixedCropDataset([(path, 1)], ["source/train/audio"], training=True)[0]
    center, _ = load_audio_segment(path, seconds=10, random_crop=False)
    w, c, label = item["views"][item["view_rows"][0]]
    assert label == 1
    assert torch.equal(w, resample_audio(center, rate, 16000))
    assert torch.equal(c, resample_audio(center, rate, 48000))
    if frames < 80000:
        assert item["view_rows"] == [0, 0, 0] and len(item["views"]) == 1
    else:
        assert len(item["view_rows"]) == 3 and len(w) == 160000 and len(c) == 480000


def test_grouping_never_pads_short_waveforms_and_preserves_row_indices():
    def view(length):
        return {"wavlm": {"input_values": torch.arange(length)[None].float(),
                          "attention_mask": torch.ones(1, length, dtype=torch.long)},
                "clap": {"input_features": torch.ones(1, 2)}, "labels": torch.tensor([0])}
    records = [{"views": [view(5), view(3), view(5)]}]
    groups = list(grouped_views(records, 2))
    assert groups[0][0] == [0, 2] and groups[0][1]["wavlm"]["input_values"].shape == (2, 5)
    assert groups[1][0] == [1] and groups[1][1]["wavlm"]["input_values"].shape == (1, 3)
    records[0]["views"][0]["wavlm"]["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="not pad"):
        list(grouped_views(records, 2))


class Projection(nn.Module):
    def forward(self, values):
        return values, values + 1000


class FakeWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = Projection()

    def forward(self, input_values, attention_mask, **kwargs):
        frontend, _ = self.feature_projection(input_values[..., None].expand(-1, -1, 768))
        # Deliberately distinct hidden_states[0] to catch the wrong R0 choice.
        states = (frontend + 1000, *(frontend + i for i in range(1, 13)))
        return SimpleNamespace(hidden_states=states)

    def _get_feature_vector_attention_mask(self, length, mask):
        return mask.bool()


class FakeClap(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_projection = nn.Identity()

    def get_audio_features(self, input_features):
        raw = self.audio_projection(input_features.expand(-1, 512))
        return nn.functional.normalize(raw, dim=-1)


def test_all_thirteen_masked_means_and_raw_clap_projection_are_fp32():
    model = nn.Module()
    model.wavlm, model.clap = FakeWavLM(), FakeClap()
    model.requires_grad_(False).eval()
    batch = {"wavlm": {"input_values": torch.tensor([[1., 3., 999.]]),
                       "attention_mask": torch.tensor([[1, 1, 0]])},
             "clap": {"input_features": torch.tensor([[7.]])}, "labels": torch.tensor([0])}
    result = all_layer_features(model, batch, "cpu")
    assert result["wavlm_layers"].shape == (1, 13, 768)
    np.testing.assert_array_equal(result["wavlm_layers"][0, :, 0], np.arange(2, 15))
    np.testing.assert_array_equal(result["clap_projected"], np.full((1, 512), 7, np.float32))
    assert all(value.dtype == np.float32 for value in result.values())
    assert not model.wavlm.feature_projection._forward_hooks and not model.clap.audio_projection._forward_hooks
    model.wavlm.train()
    with pytest.raises(ValueError, match="eval"):
        all_layer_features(model, batch, "cpu")


def test_clap_repeatpad_repeats_complete_copies_then_zero_pads(monkeypatch):
    extractor = ClapFeatureExtractor()
    captured = []
    monkeypatch.setattr(extractor, "_np_extract_fbank_features",
                        lambda waveform, *_: captured.append(waveform.copy()) or np.zeros((1001, 64)))
    signal = np.arange(70000, dtype=np.float32) / 70000
    extractor(signal, sampling_rate=48000, padding="repeatpad", truncation="rand_trunc")
    expected = np.pad(np.tile(signal, 6), (0, 60000))
    np.testing.assert_array_equal(captured[0], expected)


def test_entrypoint_reads_only_canonical_train_and_dev(monkeypatch):
    monkeypatch.setattr("scripts.lacf.cache_diagnostics.COUNTS", {key: (4, 2) for key in COUNTS})
    calls = []
    def fake_reader(domain, split, root):
        calls.append((domain, split))
        return [(f"/{domain}/{split}/{i}", i % 2) for i in range(2 if split == "dev" else 4)]
    monkeypatch.setattr("scripts.lacf.cache_diagnostics.read_dataset", fake_reader)
    monkeypatch.setattr("scripts.lacf.cache_diagnostics.utterance_id", lambda path, *args: path)
    values = source_records("/synthetic")
    assert len(values) == 8 and len(calls) == 8
    assert {split for _, split in calls} == {"train", "dev"}


def test_separate_encoder_batches_keep_canonical_feature_row_order(monkeypatch):
    def view(value, length):
        return {"wavlm": {"input_values": torch.full((1, length), float(value)),
                          "attention_mask": torch.ones(1, length, dtype=torch.long)},
                "clap": {"input_features": torch.tensor([[float(value)]])}, "labels": torch.tensor([0])}
    records = [{"views": [view(1, 3), view(2, 5)]}, {"views": [view(3, 3), view(4, 5), view(5, 3)]}]
    calls = []
    def w_features(_encoder, inputs, _device):
        calls.append(("w", tuple(inputs["input_values"].shape)))
        return inputs["input_values"][:, :1, None].expand(-1, 13, 768).numpy()
    def c_features(_encoder, inputs, _device):
        calls.append(("c", len(inputs["input_features"])))
        return inputs["input_features"].expand(-1, 512).numpy()
    monkeypatch.setattr("scripts.lacf.cache_diagnostics.pooled_wavlm_layers", w_features)
    monkeypatch.setattr("scripts.lacf.cache_diagnostics.projected_clap_audio", c_features)
    values = encode_chunk(SimpleNamespace(wavlm=None, clap=None), records, 2, 3)
    np.testing.assert_array_equal(values["wavlm_layers"][:, 0, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(values["clap_projected"][:, 0], [1, 2, 3, 4, 5])
    assert [row for row in calls if row[0] == "c"] == [("c", 3), ("c", 2)]
    assert ("w", (2, 3)) in calls and ("w", (2, 5)) in calls


def test_feature_continuation_preserves_and_verifies_shards(tmp_path):
    import json
    chunk = tmp_path / "chunk_00000"
    chunk.mkdir()
    records, names = [("/synthetic.wav", 1)], ["source/train/synthetic.wav"]
    intervals, aliases = crop_intervals(8000, 16000, names[0], training=True)
    metadata = [{"index": 0, "id": names[0], "label": 1, "rate": 16000,
                 "frames": 8000, "intervals": intervals, "view_rows": aliases}]
    (chunk / "records.json").write_text(json.dumps(metadata))
    np.save(chunk / "wavlm_layers.npy", np.ones((1, 13, 768), np.float32))
    np.save(chunk / "clap_projected.npy", np.ones((1, 512), np.float32))
    hashes = {name: checkpoint_digest(chunk / name) for name in ("wavlm_layers.npy", "clap_projected.npy", "records.json")}
    (chunk / "complete.json").write_text(json.dumps({"records": 1, "unique_rows": 1, "files_sha256": hashes}))
    before = {p.name: p.read_bytes() for p in chunk.iterdir()}
    chunks, done, unique = preserved_chunks(tmp_path, records, names, training=True)
    assert (done, unique) == (1, 1)
    assert {p.name: p.read_bytes() for p in chunk.iterdir()} == before
    completed = {"chunks": chunks, "records": 1, "unique_rows": 1}
    assert preserved_chunks(tmp_path, records, names, training=True, completed=completed) == (chunks, 1, 1)
    with pytest.raises(ValueError, match="identity"):
        preserved_chunks(tmp_path, [("/synthetic.wav", 0)], names, training=True)
    (chunk / "clap_projected.npy").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        preserved_chunks(tmp_path, records, names, training=True)
    (chunk / "clap_projected.npy").write_bytes(before["clap_projected.npy"])
    (tmp_path / "chunk_00001").mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        preserved_chunks(tmp_path, records, names, training=True)


def test_continuation_normalizes_json_keys_but_rejects_changed_revision_or_crop():
    import json
    identity = {"processors": {"sampling_rate": 16000}, "pretrained": {"revision": "fixed", "id2label": {0: "BF", 1: "spoof"}},
                "precision": "BF16", "views": {"train": ["center", "random_0", "random_1"]}, "crop_seed": 1234, "crop_rng": "fixed"}
    original = json.loads(json.dumps(identity))
    verify_scientific_identity(original, identity)
    identity["pretrained"]["revision"] = "changed"
    with pytest.raises(ValueError, match="pretrained"):
        verify_scientific_identity(original, identity)
    identity["pretrained"]["revision"] = "fixed"
    identity["crop_seed"] = 1235
    with pytest.raises(ValueError, match="crop_seed"):
        verify_scientific_identity(original, identity)
