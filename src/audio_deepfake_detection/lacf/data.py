"""One original physical crop, then separate 16/48 kHz views and balanced sampling."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from audio_deepfake_detection.data import load_audio_segment, resample_audio


class LACFDataset(Dataset):
    def __init__(self, records, *, training, segment_seconds=10):
        if not records or any(label not in (0, 1) for _, label in records):
            raise ValueError("Expected nonempty canonical binary records")
        self.records = list(records)
        self.training = training
        if segment_seconds not in (4, 10):
            raise ValueError("LACF supports the prescribed 4 s and 10 s studies")
        self.segment_seconds = segment_seconds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        # Random train / centered dev; short inputs retain their complete available signal.
        common, rate = load_audio_segment(path, seconds=self.segment_seconds, random_crop=self.training)
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


def _initialize_worker(_):
    # This host cannot use NNPACK; avoid repeating its initialization warning.
    torch.backends.nnpack.set_flags(False)


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
                      generator=torch.Generator().manual_seed(seed + 1),
                      worker_init_fn=_initialize_worker, **options)
