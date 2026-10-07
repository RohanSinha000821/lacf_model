"""One original physical crop, then separate 16/48 kHz views and balanced sampling."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from audio_deepfake_detection.data import load_audio_segment, resample_audio


class LACFDataset(Dataset):
    def __init__(self, records, *, training):
        if not records or any(label not in (0, 1) for _, label in records):
            raise ValueError("Expected nonempty canonical binary records")
        self.records = list(records)
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        # Random train / centered dev; short inputs retain their complete available signal.
        common, rate = load_audio_segment(path, seconds=10.0, random_crop=self.training)
        if not torch.isfinite(common).all():
            raise ValueError(f"Nonfinite audio: {path}")
        return (resample_audio(common, rate, 16000).float(),
                resample_audio(common, rate, 48000).float(), label)


def balanced_sampler(datasets, seed):
    if not datasets:
        raise ValueError("No source datasets")
    weights = []
    for dataset in datasets:
        labels = [label for _, label in dataset.records]
        counts = [labels.count(label) for label in (0, 1)]
        if min(counts) == 0:
            raise ValueError("Every source domain must contain both classes for balanced sampling")
        weights.extend(1 / counts[label] for label in labels)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def make_loader(datasets, collate, *, training, seed, batch_size, num_workers, prefetch_factor=2):
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("Invalid loader settings")
    options = {}
    if training:
        options["sampler"] = balanced_sampler(datasets, seed)
        dataset = ConcatDataset(datasets)
    else:
        if len(datasets) != 1 or batch_size != 1:
            raise ValueError("Development uses one source domain, batch one")
        dataset = datasets[0]
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=num_workers,
                      pin_memory=True, drop_last=False, shuffle=False,
                      generator=torch.Generator().manual_seed(seed + 1), **options)
