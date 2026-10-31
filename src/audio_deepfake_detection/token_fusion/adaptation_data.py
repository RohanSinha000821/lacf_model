"""Canonical source-only sampling of the existing three fixed physical views."""

import hashlib
import json

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from audio_deepfake_detection.data import resample_audio
from audio_deepfake_detection.lacf.data import _initialize_worker, balanced_sampler
from audio_deepfake_detection.lacf.diagnostic_cache import crop_intervals
from audio_deepfake_detection.protocol import FOLDS, read_dataset, utterance_id


class SelectedCropDataset(Dataset):
    def __init__(self, records, identifiers, *, training):
        if len(records) != len(identifiers) or not records or len(set(identifiers)) != len(records):
            raise ValueError("Expected unique canonical source records")
        self.records, self.identifiers, self.training = records, identifiers, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, selection):
        index, logical_view = selection if isinstance(selection, tuple) else (selection, 0)
        if not 0 <= logical_view < (3 if self.training else 1):
            raise ValueError("Invalid fixed logical view")
        path, label = self.records[index]
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            intervals, aliases = crop_intervals(len(stream), rate, self.identifiers[index], training=self.training)
            start, length = intervals[aliases[logical_view]]
            stream.seek(start)
            audio = stream.read(length, dtype="float32", always_2d=True)
        if len(audio) != length or not np.isfinite(audio).all() or label not in (0, 1):
            raise ValueError(f"Incomplete/nonfinite source audio: {path}")
        common = torch.from_numpy(audio).mean(1).contiguous()
        return resample_audio(common, rate, 16000).float(), resample_audio(common, rate, 48000).float(), label


class ViewSampler(Sampler):
    def __init__(self, domains, seed):
        self.sampler = balanced_sampler(domains, seed)
        self.views = torch.Generator().manual_seed(seed+1)

    def __len__(self):
        return self.sampler.num_samples

    def __iter__(self):
        order = torch.multinomial(self.sampler.weights, len(self), replacement=True, generator=self.sampler.generator)
        views = torch.randint(3, (len(self),), generator=self.views)
        return iter(zip(order.tolist(), views.tolist()))


def inner_sources(data_root, held):
    if held not in FOLDS["f1"]["sources"]:
        raise ValueError("Only F1 source inner rotations; targets remain held")
    datasets = {}
    for domain in FOLDS["f1"]["sources"]:
        part = "dev" if domain == held else "train"
        records = read_dataset(domain, part, data_root)
        ids = [f"{domain}/{part}/{utterance_id(p,domain,data_root)}" for p, _ in records]
        datasets[domain] = SelectedCropDataset(records, ids, training=part == "train")
    fitting = [datasets[d] for d in FOLDS["f1"]["sources"] if d != held]
    combined = SelectedCropDataset([r for d in fitting for r in d.records],
                                    [i for d in fitting for i in d.identifiers], training=True)
    return fitting, combined, datasets[held]


def make_audio_loader(dataset, collator, *, batch_size, workers, seed, domains=None):
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid source loader")
    sampler = ViewSampler(domains, seed) if domains is not None else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False,
                      collate_fn=collator, num_workers=workers, pin_memory=True,
                      worker_init_fn=_initialize_worker, generator=torch.Generator().manual_seed(seed+1),
                      **({"prefetch_factor": 2, "persistent_workers": True} if workers else {}))


def verify_source_membership(datasets, marker):
    """Cheap identity check against the existing cache, without reading features."""
    result = {}
    for dataset in datasets:
        domain, part, _ = dataset.identifiers[0].split("/", 2)
        prefix = f"{domain}/{part}/"
        if any(not name.startswith(prefix) for name in dataset.identifiers):
            raise ValueError("Mixed canonical source partition")
        pairs = [(name[len(prefix):], label) for name, (_, label) in zip(dataset.identifiers, dataset.records)]
        digest = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
        expected = marker["splits"][f"{domain}/{part}"]
        if len(dataset) != expected["records"] or digest != expected["canonical_ids_labels_sha256"]:
            raise ValueError(f"Source membership differs from cached reference: {domain}/{part}")
        result[f"{domain}/{part}"] = {"records": len(dataset), "canonical_ids_labels_sha256": digest}
    return result
