"""Reuse the TF2 adaptation math; load only CLAP cache payloads, never WavLM arrays."""

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio_deepfake_detection.data import resample_audio
from audio_deepfake_detection.lacf.data import _initialize_worker
from audio_deepfake_detection.lacf.diagnostic_cache import crop_intervals
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.token_fusion.adaptation import AdaptationConfig, AdaptedTF2
from audio_deepfake_detection.token_fusion.adaptation_data import ViewSampler


class CachedCLAPTF2(AdaptedTF2):
    def __init__(self, wavlm, *, fusion=None):
        # Identity has no weights or execution cost; the CLAP encoder is never loaded.
        super().__init__(wavlm, nn.Identity(), AdaptationConfig(clap_final_stage=False), fusion=fusion)

    def clap_features(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 512 or not torch.isfinite(inputs).all():
            raise ValueError("Expected finite cached raw CLAP projections [batch,512]")
        return inputs.detach().float()

    def configuration(self):
        config = super().configuration()
        if isinstance(config["fusion"], dict):
            config["fusion"] = {**config["fusion"], "wavlm": "13 live validity-pooled 768-D representations"}
        return {**config, "clap_scope": "frozen official audio projections from source cache",
                "clap_encoder_loaded": False, "wavlm_features_cached": False}


@dataclass
class CLAPSplit:
    domain: str
    part: str
    records: list
    ids: list
    metadata: list
    view_rows: torch.Tensor
    features: torch.Tensor


def cache_identity(root):
    root = Path(root)
    manifest = json.loads((root / "complete.json").read_text())
    execution = json.loads((root / "execution_accelerated.json").read_text())
    if (manifest.get("step") != 2
            or checkpoint_digest(root / "config.json") != manifest["config_sha256"]
            or checkpoint_digest(root / "execution_accelerated.json") != manifest["accelerated_execution_sha256"]):
        raise ValueError("Frozen source-cache identity changed")
    return manifest, execution


def load_clap_split(root, domain, part, manifest, data_root):
    if domain not in FOLDS["f1"]["sources"] or part not in ("train", "dev"):
        raise ValueError("Only full-F1 source train/dev partitions are allowed")
    root = Path(root)
    records = read_dataset(domain, part, data_root)
    ids = [utterance_id(path, domain, data_root) for path, _ in records]
    labels = [label for _, label in records]
    membership = hashlib.sha256(json.dumps(list(zip(ids, labels))).encode()).hexdigest()
    if (not records or len(set(ids)) != len(ids) or manifest["records"] != len(records)
            or membership != manifest["canonical_ids_labels_sha256"]
            or manifest != json.loads((root/domain/part/"complete.json").read_text())):
        raise ValueError("Canonical source cache membership changed")
    views = 3 if part == "train" else 1
    if manifest["logical_views"] != len(records)*views:
        raise ValueError("Logical crop count changed")
    features = torch.empty(manifest["unique_rows"], 512, dtype=torch.float32)
    aliases = torch.empty(len(records), views, dtype=torch.long)
    metadata, record_offset, feature_offset = [], 0, 0
    for index, descriptor in enumerate(manifest["chunks"]):
        if descriptor["path"] != f"chunk_{index:05d}":
            raise ValueError("Nonsequential feature shards")
        directory = root/domain/part/descriptor["path"]
        if checkpoint_digest(directory/"complete.json") != descriptor["completion_sha256"]:
            raise ValueError("CLAP shard completion changed")
        marker = json.loads((directory/"complete.json").read_text())
        if any(marker[key] != descriptor[key] for key in ("records", "unique_rows")):
            raise ValueError("CLAP shard counts changed")
        # WavLM arrays are deliberately not read or hashed: they are not inputs here.
        for name in ("records.json", "clap_projected.npy"):
            if checkpoint_digest(directory/name) != marker["files_sha256"][name]:
                raise ValueError("CLAP cache payload checksum mismatch")
        rows = json.loads((directory/"records.json").read_text())
        if len(rows) != marker["records"]:
            raise ValueError("CLAP shard metadata count changed")
        local_offset = 0
        for row in rows:
            intervals, mapping = crop_intervals(row["frames"], row["rate"], row["id"], training=part == "train")
            expected_aliases = [local_offset + value for value in mapping]
            if (record_offset >= len(records) or row["index"] != record_offset
                    or row["id"] != f"{domain}/{part}/{ids[record_offset]}"
                    or row["label"] != labels[record_offset]
                    or row["intervals"] != [list(value) for value in intervals]
                    or row["view_rows"] != expected_aliases):
                raise ValueError("Cached crop identity, label or aliases changed")
            aliases[record_offset] = torch.tensor(expected_aliases) + feature_offset
            metadata.append(row)
            record_offset += 1
            local_offset += len(intervals)
        array = np.load(directory/"clap_projected.npy", allow_pickle=False, mmap_mode="r")
        if (local_offset != marker["unique_rows"] or array.shape != (local_offset, 512)
                or array.dtype != np.float32 or not np.isfinite(array).all()):
            raise ValueError("Invalid CLAP payload dimensions, dtype or values")
        features[feature_offset:feature_offset+local_offset].copy_(torch.from_numpy(np.array(array)))
        feature_offset += local_offset
    if record_offset != len(records) or feature_offset != len(features):
        raise ValueError("Incomplete CLAP partition")
    return CLAPSplit(domain, part, records, ids, metadata, aliases, features)


class CachedCLAPDataset(Dataset):
    def __init__(self, splits):
        if not splits or len({split.part for split in splits}) != 1:
            raise ValueError("Expected nonempty same-part source splits")
        self.splits = splits
        self.records = [record for split in splits for record in split.records]
        self.ends = np.cumsum([len(split.records) for split in splits]).tolist()
        self.training = splits[0].part == "train"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, selection):
        index, view = selection if isinstance(selection, tuple) else (selection, 0)
        if not 0 <= index < len(self) or not 0 <= view < (3 if self.training else 1):
            raise ValueError("Invalid recording index or fixed crop view")
        group = bisect_right(self.ends, index)
        local = index - (self.ends[group-1] if group else 0)
        split = self.splits[group]
        row = split.metadata[local]
        intervals, mapping = crop_intervals(row["frames"], row["rate"], row["id"], training=self.training)
        start, length = intervals[mapping[view]]
        with sf.SoundFile(split.records[local][0]) as stream:
            if stream.samplerate != row["rate"] or len(stream) != row["frames"]:
                raise ValueError("Source audio length/rate changed since caching")
            stream.seek(start)
            audio = stream.read(length, dtype="float32", always_2d=True)
        if len(audio) != length or not np.isfinite(audio).all():
            raise ValueError("Incomplete/nonfinite source audio")
        wave = resample_audio(torch.from_numpy(audio).mean(1), row["rate"], 16000).float()
        return wave, split.features[split.view_rows[local, view]], split.records[local][1]


class CachedCLAPCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        waves, clap, labels = zip(*items)
        if any(wave.ndim != 1 or not 400 <= len(wave) <= 160000 for wave in waves):
            raise ValueError("Expected valid native WavLM views up to ten seconds")
        inputs = self.processor([wave.numpy() for wave in waves], sampling_rate=16000,
                                padding=True, return_attention_mask=True, return_tensors="pt")
        return {"wavlm": dict(inputs), "clap": torch.stack(clap), "labels": torch.tensor(labels)}


def make_loader(dataset, collator, *, batch_size, workers, seed, domains=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      sampler=ViewSampler(domains, seed) if domains is not None else None,
                      collate_fn=collator, num_workers=workers, pin_memory=True,
                      worker_init_fn=_initialize_worker, generator=torch.Generator().manual_seed(seed+1),
                      **({"persistent_workers": True, "prefetch_factor": 2} if workers else {}))
