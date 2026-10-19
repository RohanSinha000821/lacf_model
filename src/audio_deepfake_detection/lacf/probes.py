"""Small frozen-feature probes and batched source-only inputs for Step 3."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audio_deepfake_detection.lacf.data import balanced_sampler
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss, relation_classifier, unit_vectors
from audio_deepfake_detection.metrics import compute_auroc, compute_eer
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id


INPUTS = ("last", "weighted", "clap", "concat")
HEADS = ("logistic", "mlp")
SEEDS = (1234, 1235, 1236)
OUTER_FOLDS = ("f1", "f2", "f3")
BATCH_SIZE = 132


def inner_domains(fold, held):
    if fold not in OUTER_FOLDS or held not in FOLDS[fold]["sources"]:
        raise ValueError("Only a held-in-source domain of F1–F3 may validate Step 3")
    return tuple(domain for domain in FOLDS[fold]["sources"] if domain != held)


class Probe(nn.Module):
    def __init__(self, inputs, head):
        super().__init__()
        if inputs not in INPUTS or head not in HEADS:
            raise ValueError("Unknown Step 3 input/head")
        self.inputs, self.head = inputs, head
        self.layer_weights = nn.Parameter(torch.zeros(13)) if inputs in ("weighted", "concat") else None
        dimension = 512 if inputs == "clap" else 1280 if inputs == "concat" else 768
        self.classifier = nn.Linear(dimension, 1) if head == "logistic" else relation_classifier(dimension)
        self.objective = LACFLoss(ComponentConfig(semantic_loss=False, bonafide_loss=False))

    def forward(self, batch):
        values = []
        # WavLM pooled states retain the native unadapted speech representation.
        # Only the 13 mixing scalars and the chosen head are fitted.
        if self.inputs != "clap":
            with torch.autocast(batch["wavlm"].device.type, enabled=False):
                w = batch["wavlm"].float()
                if self.layer_weights is not None:
                    w = (w * self.layer_weights.float().softmax(0)[None, :, None]).sum(1)
            values.append(w)
        if self.inputs in ("clap", "concat"):
            values.append(unit_vectors(batch["clap"]))
        return {"logit": self.classifier(torch.cat(values, dim=-1)).squeeze(-1)}

    def configuration(self):
        return {"inputs": self.inputs, "head": self.head,
                "input_dim": 512 if self.inputs == "clap" else 1280 if self.inputs == "concat" else 768,
                "wavlm": "raw masked mean; no adapter or extra feature standardization",
                "layer_mix": "fresh uniform softmax over projected frontend plus twelve Transformer outputs"
                             if self.layer_weights is not None else None,
                "clap": "L2-normalized projected embedding; FP32 normalization",
                "mlp": "LN(input)→64→GELU→dropout(.2)→LN(64)→16→GELU→dropout(.2)→1" if self.head == "mlp" else None,
                "objective": "detection BCE only; no anchors or auxiliary losses",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


@dataclass
class FeatureSplit:
    records: list
    ids: list[str]
    labels: torch.Tensor
    view_rows: torch.Tensor
    wavlm: torch.Tensor
    clap: torch.Tensor


def load_split(root, domain, split, manifest, data_root, *, logger=None, last_layer_only=False):
    """Validate once and load compact CPU arrays, avoiding per-example disk reads."""
    if domain not in FOLDS["f1"]["sources"] + ("speechfake",) or split not in ("train", "dev"):
        raise ValueError("Probe cache access is restricted to canonical source train/dev")
    root = Path(root)
    records = read_dataset(domain, split, data_root)
    ids = [utterance_id(path, domain, data_root) for path, _ in records]
    labels = torch.tensor([label for _, label in records], dtype=torch.long)
    membership = hashlib.sha256(json.dumps(list(zip(ids, labels.tolist()))).encode()).hexdigest()
    if (manifest["records"] != len(records) or len(set(ids)) != len(records)
            or membership != manifest["canonical_ids_labels_sha256"]
            or manifest != json.loads((root / domain / split / "complete.json").read_text())):
        raise ValueError(f"Canonical cache membership changed: {domain}/{split}")
    count = manifest["unique_rows"]
    wavlm = torch.empty((count, 768) if last_layer_only else (count, 13, 768), dtype=torch.float32)
    clap = torch.empty(count, 512, dtype=torch.float32)
    view_rows = torch.empty(len(records), 3 if split == "train" else 1, dtype=torch.long)
    record_offset = feature_offset = 0
    for index, descriptor in enumerate(manifest["chunks"]):
        if descriptor["path"] != f"chunk_{index:05d}":
            raise ValueError("Nonsequential feature shards")
        directory = root / domain / split / descriptor["path"]
        if checkpoint_digest(directory / "complete.json") != descriptor["completion_sha256"]:
            raise ValueError("Feature shard manifest changed")
        marker = json.loads((directory / "complete.json").read_text())
        if marker["records"] != descriptor["records"] or marker["unique_rows"] != descriptor["unique_rows"]:
            raise ValueError("Feature shard counts changed")
        for name in ("records.json", "wavlm_layers.npy", "clap_projected.npy"):
            if checkpoint_digest(directory / name) != marker["files_sha256"][name]:
                raise ValueError(f"Feature payload checksum mismatch: {directory / name}")
        metadata = json.loads((directory / "records.json").read_text())
        if len(metadata) != marker["records"]:
            raise ValueError("Feature shard record count changed")
        for row in metadata:
            if (record_offset >= len(records) or row["index"] != record_offset
                    or row["id"] != f"{domain}/{split}/{ids[record_offset]}" or row["label"] != labels[record_offset].item()
                    or len(row["view_rows"]) != view_rows.shape[1]
                    or any(not 0 <= value < marker["unique_rows"] for value in row["view_rows"])):
                raise ValueError("Feature identity, labels or crop aliases changed")
            view_rows[record_offset] = torch.tensor(row["view_rows"]) + feature_offset
            record_offset += 1
        end = feature_offset + marker["unique_rows"]
        for filename, destination in (("wavlm_layers.npy", wavlm), ("clap_projected.npy", clap)):
            array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
            shape = (marker["unique_rows"], 13, 768) if filename == "wavlm_layers.npy" else (marker["unique_rows"], 512)
            if array.dtype != np.float32 or array.shape != shape:
                raise ValueError("Invalid feature dtype or dimensions")
            if filename == "wavlm_layers.npy" and last_layer_only:
                array = array[:, -1]
            np.copyto(destination.numpy()[feature_offset:end], array)
        feature_offset = end
        if logger is not None and ((index + 1) % 100 == 0 or index + 1 == len(manifest["chunks"])):
            logger.info("Cache %s/%s: verified and loaded %d/%d records", domain, split, record_offset, len(records))
    if record_offset != len(records) or feature_offset != count:
        raise ValueError("Incomplete feature cache")
    return FeatureSplit(records, ids, labels, view_rows, wavlm, clap)


def load_bank(root, data_root, *, logger=None, last_layer_only=False):
    root = Path(root)
    manifest = json.loads((root / "complete.json").read_text())
    if (manifest.get("step") != 2 or checkpoint_digest(root / "config.json") != manifest["config_sha256"]
            or checkpoint_digest(root / "execution_accelerated.json") != manifest["accelerated_execution_sha256"]):
        raise ValueError("Step 2 completion/configuration identity changed")
    domains = tuple(dict.fromkeys(domain for fold in OUTER_FOLDS for domain in FOLDS[fold]["sources"]))
    if set(manifest["splits"]) != {f"{d}/{s}" for d in domains for s in ("train", "dev")}:
        raise ValueError("Missing/unexpected source cache partitions")
    return {(d, s): load_split(root, d, s, manifest["splits"][f"{d}/{s}"], data_root, logger=logger,
                              last_layer_only=last_layer_only)
            for d in domains for s in ("train", "dev")}


class FeatureBatches:
    """Vectorized utterance sampling and feature gathers; no worker/file per item."""
    def __init__(self, datasets, inputs, device, *, training, seed, batch_size=BATCH_SIZE, max_batches=None,
                 last_layer_only=False):
        if not datasets or inputs not in INPUTS or batch_size < 1 or max_batches is not None and max_batches < 1:
            raise ValueError("Invalid feature batches")
        if not training and len(datasets) != 1:
            raise ValueError("Score one inner validation domain at a time")
        self.datasets, self.inputs, self.device = datasets, inputs, torch.device(device)
        if last_layer_only and inputs not in ("last", "concat"):
            raise ValueError("Compact last-layer batches cannot fit a learned layer mixture")
        self.last_layer_only = inputs == "last" or last_layer_only
        if any(d.wavlm.ndim == 2 for d in datasets) and inputs != "clap" and not self.last_layer_only:
            raise ValueError("A learned mixture requires all thirteen cached layers")
        self.training, self.batch_size = training, batch_size
        self.offsets = np.cumsum([0] + [len(d.labels) for d in datasets]).tolist()
        self.count = self.offsets[-1] if max_batches is None else min(self.offsets[-1], batch_size * max_batches)
        self.sampler = balanced_sampler(datasets, seed) if training else None
        self.view_generator = torch.Generator().manual_seed(seed + 1)
        shapes = {"labels": (batch_size,)}
        if inputs != "clap":
            shapes["wavlm"] = (batch_size, 768) if self.last_layer_only else (batch_size, 13, 768)
        if inputs in ("clap", "concat"):
            shapes["clap"] = (batch_size, 512)
        self.buffers = {key: torch.empty(shape, dtype=torch.long if key == "labels" else torch.float32,
                                         pin_memory=self.device.type == "cuda") for key, shape in shapes.items()}

    def __len__(self):
        return math.ceil(self.count / self.batch_size)

    def __iter__(self):
        if self.training:
            # Same generator/law as the shared WeightedRandomSampler, without a
            # million Python __getitem__ calls or tripling epoch visits for views.
            order = torch.multinomial(self.sampler.weights, self.count, replacement=True, generator=self.sampler.generator)
            columns = torch.randint(3, (self.count,), generator=self.view_generator)
        else:
            order, columns = torch.arange(self.count), torch.zeros(self.count, dtype=torch.long)
        transfer = None
        for start in range(0, self.count, self.batch_size):
            if transfer is not None:
                # Do not overwrite pinned host storage before its CUDA copy ends.
                transfer.synchronize()
            selected, views = order[start:start + self.batch_size], columns[start:start + self.batch_size]
            size = len(selected)
            for index, dataset in enumerate(self.datasets):
                positions = ((selected >= self.offsets[index]) & (selected < self.offsets[index + 1])).nonzero().flatten()
                if not len(positions):
                    continue
                records = selected[positions] - self.offsets[index]
                rows = dataset.view_rows[records, views[positions]]
                self.buffers["labels"][positions] = dataset.labels[records]
                if "wavlm" in self.buffers:
                    source = (dataset.wavlm[:, -1] if self.last_layer_only and dataset.wavlm.ndim == 3
                              else dataset.wavlm)
                    self.buffers["wavlm"][positions] = source[rows]
                if "clap" in self.buffers:
                    self.buffers["clap"][positions] = dataset.clap[rows]
            batch = {key: value[:size].to(self.device, non_blocking=True) for key, value in self.buffers.items()}
            if self.device.type == "cuda":
                transfer = torch.cuda.Event()
                transfer.record()
            yield batch


@torch.inference_mode()
def score_probe(model, loader):
    model.eval()
    scores = torch.empty(loader.count, dtype=torch.float32, device=loader.device)
    offset = 0
    for batch in loader:
        with torch.autocast(loader.device.type, dtype=torch.bfloat16):
            values = model(batch)["logit"].float()
        scores[offset:offset + len(values)] = values
        offset += len(values)
    scores = scores.cpu()
    labels = loader.datasets[0].labels[:loader.count]
    return scores, {"eer": compute_eer(labels.numpy(), scores.numpy()), "auroc": compute_auroc(labels.numpy(), scores.numpy())}
