"""LODO adapter around the licensed, unmodified authors' AASIST architecture.

Reference commit: a04c9863f63d44471dde8a6abcb3b082b07cd1d1.
The original network's output order is [spoof, bona fide]; public logits here
are [bona fide, spoof]. No released ASVspoof-trained weights are loaded.
"""

from copy import deepcopy

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from audio_deepfake_detection.data import Record, load_audio_segment, resample_audio
from audio_deepfake_detection.sota.aasist_arch import Model

UPSTREAM_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
SAMPLE_RATE = 16000
NUM_SAMPLES = 64600
MODEL_CONFIG = {
    "architecture": "AASIST", "nb_samp": NUM_SAMPLES, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def fixed_length_audio(waveform: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Random train crop / first eval crop; repeat short signals, never zero-pad.

    The released random crop fails for exactly 64,600 samples and excludes the
    last valid starting point. This version accepts exact lengths and samples
    every valid starting point. Crop selection follows resampling to 16 kHz.
    """
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Expected a nonempty, finite mono waveform")
    length = waveform.numel()
    if length < NUM_SAMPLES:
        return waveform.repeat((NUM_SAMPLES + length - 1) // length)[:NUM_SAMPLES]
    start = int(torch.randint(length - NUM_SAMPLES + 1, (1,)).item()) if training else 0
    return waveform[start:start + NUM_SAMPLES].contiguous()


class AASISTDataset(Dataset):
    def __init__(self, records: list[Record], *, training: bool):
        if not records:
            raise ValueError("Dataset contains no records")
        self.records = list(records)
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        waveform, rate = load_audio_segment(path)
        waveform = resample_audio(waveform, rate, SAMPLE_RATE).float()
        return fixed_length_audio(waveform, training=self.training), label


def make_train_loader(datasets, *, batch_size=24, seed=1234, num_workers=8, prefetch_factor=2):
    """Equal domain probability, natural within-domain classes, N replacement draws."""
    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("All source datasets must be nonempty")
    weights = torch.cat([torch.full((len(dataset),), 1 / len(dataset), dtype=torch.double)
                         for dataset in datasets])
    sampler = WeightedRandomSampler(weights, sum(map(len, datasets)), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return _loader(ConcatDataset(datasets), batch_size=batch_size, num_workers=num_workers,
                   prefetch_factor=prefetch_factor, sampler=sampler,
                   generator=torch.Generator().manual_seed(seed + 1))


def make_eval_loader(dataset, *, num_workers=4, prefetch_factor=2):
    # Batch one until numerical equivalence of larger inference batches is validated.
    return _loader(dataset, batch_size=1, num_workers=num_workers, prefetch_factor=prefetch_factor)


def _loader(dataset, *, batch_size, num_workers, prefetch_factor, **kwargs):
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("Invalid loader settings")
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, shuffle=False, drop_last=False, **kwargs)


class AASIST(Model):
    def __init__(self):
        super().__init__(deepcopy(MODEL_CONFIG))

    def forward(self, waveforms):
        if waveforms.ndim != 2 or waveforms.shape[1] != NUM_SAMPLES:
            raise ValueError(f"Expected [batch, {NUM_SAMPLES}] waveforms")
        # Released AASIST.conf defaults freq_aug=False; do not add RawBoost here.
        _, original_logits = super().forward(waveforms, Freq_aug=False)
        return original_logits[:, [1, 0]]

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]
