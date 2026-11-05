"""Separate last-four-block SSL adaptation; the original full-FT baseline is unchanged."""

import torch

from audio_deepfake_detection.sota.ssl_gating import SSLFeatureGating, MODEL_CONFIG, cka_loss
from audio_deepfake_detection.sota.ssl_gating_blocks import MultiHeadPooling


RUN_FAMILY = "ssl_gating_20s_top4_bf16"
PARTIAL_MODEL_CONFIG = {
    **MODEL_CONFIG, "adaptation_version": 3,
    "fine_tuning": "last_four_xlsr_transformer_blocks_and_complete_gating_backend",
    "trainable_encoder_blocks": [20, 21, 22, 23],
    "frozen_prefix_stochasticity": "original_training_specaugment_dropout_layerdrop_retained",
    "precision": "bfloat16_autocast_float32_pooling_cka_and_ce",
}


class FP32Pooling(MultiHeadPooling):
    def forward(self, values):
        with torch.autocast(values.device.type, enabled=False):
            return super().forward(values.float())


class PartialSSLFeatureGating(SSLFeatureGating):
    def __init__(self, feature_extractor=None):
        super().__init__(feature_extractor)
        self.feature_extractor.requires_grad_(False)
        # HF otherwise forces input gradients in train mode even with frozen weights.
        self.feature_extractor.feature_extractor._freeze_parameters()
        for block in self.feature_extractor.encoder.layers[-4:]:
            block.requires_grad_(True)
        original_pooling = self.pooling.state_dict()
        with torch.random.fork_rng(devices=[]):
            self.pooling = FP32Pooling()
        self.pooling.load_state_dict(original_pooling)


def detection_losses(model, waves, labels):
    with torch.autocast(waves.device.type, dtype=torch.bfloat16):
        logits, means = model(waves, return_block_means=True)
    with torch.autocast(waves.device.type, enabled=False):
        ce = torch.nn.functional.cross_entropy(
            logits.float(), labels, weight=torch.tensor([0.9, 0.1], device=waves.device))
        cka = cka_loss(means.float())
    return ce + cka, ce, cka


@torch.inference_mode()
def partial_scores(model, waves):
    with torch.autocast(waves.device.type, dtype=torch.bfloat16):
        logits = model(waves)
    return model.spoof_score(logits.float())
