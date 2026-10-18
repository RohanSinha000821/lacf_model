"""Full-utterance XLS-R feature gating, adapted to the source-only LODO study.

The released block/time reshape and detached CKA are repaired explicitly.
No RawBoost, waveform normalization or padding masks are added. The pretrained
XLS-R's own SpecAugment/dropout/layerdrop settings remain active in training.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from audio_deepfake_detection.data import Record, load_audio_segment, resample_audio
from audio_deepfake_detection.sota.ssl_gating_blocks import MultiConv, MultiHeadPooling

UPSTREAM_COMMIT = "4de821cad840ef6b4f78905f9cce0523c956d988"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m"
PRETRAINED_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
SAMPLE_RATE = 16000
MODEL_CONFIG = {
    "architecture": "SSLFeatureGating", "adaptation_version": 1,
    "pretrained_model": PRETRAINED_MODEL, "pretrained_revision": PRETRAINED_REVISION,
    "hidden_states": 25, "hidden_size": 1024, "projection": 128,
    "blocks": 4, "linear_units": 1024, "kernels": [3, 7, 11, 15],
    "merge_kernel": 15, "dropout": 0.1, "layer_norm_eps": 1e-12,
    "attention_heads": 4, "attention_implementation": "sdpa",
    "feature_layout": "per_time_block_concatenation", "cka": "differentiable_biased_linear",
    "rawboost": False, "input_policy": "16kHz_full_utterance_no_normalization_no_mask",
}


class FullAudioDataset(Dataset):
    def __init__(self, records: list[Record]):
        if not records:
            raise ValueError("Dataset contains no records")
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        waveform, rate = load_audio_segment(path)
        waveform = resample_audio(waveform, rate, SAMPLE_RATE).float()
        if waveform.ndim != 1 or not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError(f"Expected nonempty finite mono audio: {path}")
        if label not in (0, 1):
            raise ValueError(f"Invalid label: {label}")
        return waveform, label


def collate_audio(batch):
    waves, labels = zip(*batch)
    return pad_sequence(waves, batch_first=True, padding_value=0.0), torch.tensor(labels, dtype=torch.long)


def _loader(dataset, *, batch_size, num_workers, prefetch_factor, **kwargs):
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("Invalid loader settings")
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_audio, shuffle=False, drop_last=False, **kwargs)


def make_train_loader(datasets, *, batch_size=5, seed=1234, num_workers=16, prefetch_factor=2):
    """Equal domain probability, natural classes, N replacement draws per epoch."""
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
    return _loader(dataset, batch_size=1, num_workers=num_workers, prefetch_factor=prefetch_factor)


def linear_cka(a, b):
    """Biased linear CKA across batch examples; keep its gradient on-device.

    Centered Gram inner products equal the released trace(K H L H). The common
    HSIC normalization cancels. Zero-variance/single-example batches return 0;
    clamping before sqrt also prevents undefined zero-denominator gradients.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or not a.shape[0]:
        raise ValueError("CKA expects two nonempty [batch, features] matrices")
    a, b = a.float(), b.float()
    a, b = a - a.mean(dim=0), b - b.mean(dim=0)
    k, l = a @ a.T, b @ b.T
    numerator = (k * l).sum()
    product = k.square().sum() * l.square().sum()
    denominator = product.clamp_min(torch.finfo(product.dtype).tiny).sqrt() + 1e-6
    return numerator / denominator


def cka_loss(block_means):
    """Average absolute CKA over the six distinct block pairs, [B,4,128]."""
    if block_means.ndim != 3 or block_means.shape[1:] != (4, 128):
        raise ValueError("Expected four pooled block means of dimension 128")
    return torch.stack([linear_cka(block_means[:, i], block_means[:, j]).abs()
                        for i in range(4) for j in range(i + 1, 4)]).mean()


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(128, 128)
        self.linear_2 = nn.Linear(128, 128)

    def forward(self, x):
        # Released SwiGLU uses SiLU, despite the paper's sigmoid-only equation.
        return torch.nn.functional.silu(self.linear_1(x)) * self.linear_2(x)


class SSLFeatureGating(nn.Module):
    def __init__(self, feature_extractor=None):
        super().__init__()
        if feature_extractor is None:
            from transformers import Wav2Vec2Model
            feature_extractor = Wav2Vec2Model.from_pretrained(
                PRETRAINED_MODEL, revision=PRETRAINED_REVISION, local_files_only=True,
                attn_implementation=MODEL_CONFIG["attention_implementation"])
        if (feature_extractor.config.hidden_size != 1024
                or feature_extractor.config.num_hidden_layers != 24):
            raise ValueError("Expected the XLS-R 300M 24-layer, 1024-dimensional frontend")
        self.feature_extractor = feature_extractor
        self.blocks = nn.ModuleList([MultiConv() for _ in range(4)])
        self.feature_projection = nn.Linear(1024, 128)
        self.pooling = MultiHeadPooling()
        self.gating = SwiGLU()
        self.classifier = nn.Sequential(nn.Linear(1024, 512), nn.SELU(), nn.Linear(512, 2))

    @classmethod
    def from_frontend_config(cls, frontend_config):
        """Reconstruct for strict checkpoint reload, without pretrained downloads."""
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        config = Wav2Vec2Config.from_dict(frontend_config)
        config._attn_implementation = MODEL_CONFIG["attention_implementation"]
        return cls(Wav2Vec2Model(config))

    def forward(self, waveforms, *, return_block_means=False):
        if waveforms.ndim != 2 or waveforms.shape[1] < 400:
            raise ValueError("Expected [batch,samples] audio with at least 400 samples")
        hidden = self.feature_extractor(waveforms, output_hidden_states=True).hidden_states
        if len(hidden) != 25:
            raise ValueError("Frontend must return all 25 hidden states")
        # Project before stacking to avoid materializing a second B*25*T*1024
        # tensor. These shared, pointwise operations are algebraically identical.
        levels = [self.gating(self.feature_projection(level)) for level in hidden]
        x = torch.stack(levels, dim=1).sum(dim=1)
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        # Upstream stack(...).view(B,T,-1) interleaves block/time positions.
        pooled = self.pooling(torch.cat(outputs, dim=-1))
        logits = self.classifier(pooled)
        # Released class order is [spoof,bona fide]. Avoid list-based CUDA indexing.
        logits = logits.flip(1)
        if return_block_means:
            return logits, pooled[:, :512].reshape(-1, 4, 128)
        return logits

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]
