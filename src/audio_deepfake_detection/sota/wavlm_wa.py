from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from transformers import WavLMModel

from audio_deepfake_detection.data import (
    Record,
    load_audio_segment,
    resample_audio,
)


# =========================================================
# WavLM input configuration
# =========================================================

WAVLM_SAMPLE_RATE = 16_000
WAVLM_TRAIN_SECONDS = 4.0
WAVLM_TRAIN_SAMPLES = 64_000


# =========================================================
# WavLM dataset
# =========================================================

class WavLMDataset(Dataset):
    """
    Prepare audio for the WavLM-WA baseline.

    Training:
        random continuous 4-second physical segment
        -> 16 kHz
        -> exactly 64,000 samples

    Development / evaluation:
        full utterance
        -> 16 kHz
        -> variable length
    """

    def __init__(
        self,
        records: Sequence[Record],
        *,
        training: bool,
    ) -> None:
        if not records:
            raise ValueError("Dataset contains no records")

        self.records = list(records)
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.records[index]

        if self.training:
            waveform, source_rate = load_audio_segment(
                path,
                seconds=WAVLM_TRAIN_SECONDS,
                random_crop=True,
            )
        else:
            waveform, source_rate = load_audio_segment(
                path,
                seconds=None,
                random_crop=False,
            )

        # WavLM requires 16 kHz.
        waveform = resample_audio(
            waveform,
            source_rate=source_rate,
            target_rate=WAVLM_SAMPLE_RATE,
        ).to(dtype=torch.float32)

        if self.training:
            valid_samples = min(
                waveform.numel(),
                WAVLM_TRAIN_SAMPLES,
            )

            if waveform.numel() > WAVLM_TRAIN_SAMPLES:
                waveform = waveform[:WAVLM_TRAIN_SAMPLES]
            elif waveform.numel() < WAVLM_TRAIN_SAMPLES:
                padding = WAVLM_TRAIN_SAMPLES - waveform.numel()
                waveform = F.pad(
                    waveform,
                    (0, padding),
                    value=0.0,
                )

            attention_mask = (
                torch.arange(WAVLM_TRAIN_SAMPLES)
                < valid_samples
            )
        else:
            attention_mask = torch.ones(
                waveform.numel(),
                dtype=torch.bool,
            )

        return waveform, attention_mask, label


# =========================================================
# WavLM batch collation
# =========================================================

def collate_wavlm_batch(
    batch: Sequence[
        tuple[torch.Tensor, torch.Tensor, int]
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    waveforms = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]

    labels = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.long,
    )

    waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )

    attention_masks = pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=False,
    )

    return waveforms, attention_masks, labels


# =========================================================
# WavLM dataset-balanced sampling
# =========================================================

def make_wavlm_sampler(
    datasets: Sequence[Dataset],
    *,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """
    Sampling rule:

        dataset approximately uniformly
            ->
        utterance uniformly inside selected dataset

    Natural class ratios inside each dataset are preserved.
    """

    if not datasets:
        raise ValueError("No source datasets provided")

    lengths = [len(dataset) for dataset in datasets]

    if any(length <= 0 for length in lengths):
        raise ValueError(
            "Every source dataset must contain at least one sample"
        )

    weights: list[float] = []

    for length in lengths:
        # Each dataset receives total sampling weight 1.
        sample_weight = 1.0 / length
        weights.extend([sample_weight] * length)

    if num_samples is None:
        # One logical epoch contains the same number of draws
        # as the total number of source training examples.
        num_samples = sum(lengths)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


# =========================================================
# WavLM training DataLoader
# =========================================================

def make_wavlm_train_loader(
    datasets: Sequence[WavLMDataset],
    *,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    if not datasets:
        raise ValueError("No training datasets provided")

    combined_dataset = ConcatDataset(list(datasets))

    sampler = make_wavlm_sampler(
        datasets,
        seed=seed,
    )

    # Used by DataLoader for worker seed generation.
    generator = torch.Generator()
    generator.manual_seed(seed + 1)

    loader_kwargs = {}

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_wavlm_batch,
        generator=generator,
        **loader_kwargs,
    )


# =========================================================
# WavLM evaluation DataLoader
# =========================================================

def make_wavlm_eval_loader(
    dataset: WavLMDataset,
    *,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    loader_kwargs = {}

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_wavlm_batch,
        **loader_kwargs,
    )


# =========================================================
# WavLM-WA model
# =========================================================

class WavLMWA(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
    ) -> None:
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(model_name)

        # The baseline does not define WavLM internal SpecAugment
        # as an additional augmentation.
        self.wavlm.config.apply_spec_augment = False

        hidden_size = self.wavlm.config.hidden_size

        if hidden_size != 768:
            raise ValueError(
                f"Expected WavLM hidden size 768, got {hidden_size}"
            )

        # Projected frontend representation
        # + 12 Transformer-layer representations.
        self.layer_weights = nn.Parameter(torch.zeros(13))

        self.classifier = nn.Linear(
            hidden_size,
            2,
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def capture_frontend(
            module: nn.Module,
            inputs: tuple,
            output: tuple[torch.Tensor, ...],
        ) -> None:
            # WavLMFeatureProjection returns the projected hidden
            # representation as output[0].
            captured["frontend"] = output[0]

        handle = self.wavlm.feature_projection.register_forward_hook(
            capture_frontend
        )

        try:
            outputs = self.wavlm(
                input_values=waveforms,
                attention_mask=attention_mask.long(),
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            handle.remove()

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError(
                "WavLM did not return hidden states"
            )

        if "frontend" not in captured:
            raise RuntimeError(
                "WavLM feature_projection output was not captured"
            )

        frontend = captured["frontend"]

        # hidden_states[0] is intentionally NOT used as R0.
        # R1...R12 are the outputs of the 12 Transformer blocks.
        transformer_layers = hidden_states[1:]

        representations = (
            frontend,
            *transformer_layers,
        )

        if len(representations) != 13:
            raise RuntimeError(
                "Expected projected frontend + 12 Transformer "
                f"representations, got {len(representations)}"
            )

        # [13, B, T, 768]
        stacked = torch.stack(
            representations,
            dim=0,
        )

        # 13 learned scalar weights, normalized with softmax.
        weights = torch.softmax(
            self.layer_weights,
            dim=0,
        )

        # [B, T, 768]
        hidden = (
            stacked
            * weights[:, None, None, None]
        ).sum(dim=0)

        # Convert the raw-audio attention mask [B, samples]
        # to the WavLM feature-frame mask [B, T].
        feature_mask = (
            self.wavlm._get_feature_vector_attention_mask(
                hidden.shape[1],
                attention_mask,
            )
        )

        mask = feature_mask.unsqueeze(-1).to(
            dtype=hidden.dtype
        )

        # Masked temporal mean -> [B, 768]
        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

        # [B, 2]
        return self.classifier(pooled)

    @staticmethod
    def spoof_score(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Higher score = more spoof-like.
        return logits[:, 1] - logits[:, 0]
