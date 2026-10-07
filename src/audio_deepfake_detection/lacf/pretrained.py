"""Checkpoint processors and frozen encoders. All loads are local-only."""

import torch
from transformers import ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor, WavLMModel

from audio_deepfake_detection.lacf.model import (
    CLAP_NAME, CONCEPTS, TEMPLATES, WAVLM_NAME, LACF, ComponentConfig, text_prototypes,
)


class DualViewCollator:
    def __init__(self, wavlm_processor, clap_processor):
        self.wavlm = wavlm_processor
        self.clap = clap_processor.feature_extractor
        if (self.wavlm.sampling_rate != 16000 or self.wavlm.do_normalize
                or self.wavlm.padding_side != "right" or self.wavlm.padding_value != 0):
            raise ValueError("Unexpected WavLM Base+ processor policy")
        if (self.clap.sampling_rate != 48000 or self.clap.nb_max_samples != 480000
                or self.clap.padding != "repeatpad" or self.clap.truncation != "rand_trunc"):
            raise ValueError("Unexpected CLAP unfused processor policy")

    def configuration(self):
        return {"wavlm": self.wavlm.to_dict(), "clap": self.clap.to_dict(),
                "common_crop_seconds": 10, "train_crop": "random_native_interval",
                "dev_crop": "centered_native_interval", "wavlm_waveform_normalization": False,
                "wavlm_pooling": "last_hidden_state_feature_frame_masked_mean",
                "clap_padding": "repeatpad", "clap_secondary_random_crop": False}

    def __call__(self, items):
        if not items:
            raise ValueError("Cannot collate an empty batch")
        for w, c, _ in items:
            if (w.ndim != 1 or c.ndim != 1 or not 400 <= len(w) <= 160000
                    or not 0 < len(c) <= 480000 or not torch.isfinite(w).all() or not torch.isfinite(c).all()):
                raise ValueError("Invalid views or audio too short for WavLM; no samples are skipped")
        wavlm = self.wavlm([w.numpy() for w, _, _ in items], sampling_rate=16000,
                           padding=True, return_attention_mask=True, return_tensors="pt")
        clap = self.clap([c.numpy() for _, c, _ in items], sampling_rate=48000,
                         padding="repeatpad", truncation="rand_trunc", return_tensors="pt")
        return {"wavlm": dict(wavlm),
                "clap": {key: value.float() if value.is_floating_point() else value
                         for key, value in clap.items()},
                "labels": torch.tensor([label for _, _, label in items], dtype=torch.long)}


def load_local_model(*, config=ComponentConfig()):
    """Use already-cached original checkpoints and tokenizers, never fetch them.

    Processor revisions follow the cached model revision so a changed upstream
    processor cannot silently alter an experiment. Frozen text anchors are stored
    as a model buffer and checkpointed with the trainable state.
    """
    wavlm = WavLMModel.from_pretrained(WAVLM_NAME, local_files_only=True, dtype=torch.float32)
    clap = ClapModel.from_pretrained(CLAP_NAME, local_files_only=True, dtype=torch.float32)
    if wavlm.config.num_hidden_layers != 12 or clap.config.audio_config.enable_fusion:
        raise ValueError("Expected WavLM Base+ and unfused CLAP architecture")
    wavlm_revision = wavlm.config._commit_hash
    clap_revision = clap.config._commit_hash
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        WAVLM_NAME, revision=wavlm_revision, local_files_only=True)
    clap_processor = ClapProcessor.from_pretrained(CLAP_NAME, revision=clap_revision, local_files_only=True)
    clap.requires_grad_(False).eval()
    prompts = [template.format(concept=concept) for concept in CONCEPTS for template in TEMPLATES]
    tokens = clap_processor(text=prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        prototypes = text_prototypes(clap.get_text_features(**tokens))
    model = LACF(wavlm, clap, prototypes, config=config)
    collator = DualViewCollator(wavlm_processor, clap_processor)
    return model, collator, {"wavlm_revision": wavlm_revision, "clap_revision": clap_revision,
                            "encoder_configs": {"wavlm": wavlm.config.to_dict(), "clap": clap.config.to_dict()}}
