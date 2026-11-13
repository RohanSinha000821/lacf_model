"""Last XLS-R block plus the existing SSL-gating backend; earlier blocks frozen."""

from audio_deepfake_detection.sota.ssl_gating_partial import (
    PARTIAL_MODEL_CONFIG, PartialSSLFeatureGating,
)

RUN_FAMILY = 'ssl_gating_last'
TRAINABLE_PARAMETERS = 15_241_090
LAST_MODEL_CONFIG = {
    **PARTIAL_MODEL_CONFIG, 'adaptation_version':4,
    'fine_tuning':'last_xlsr_transformer_block_and_complete_gating_backend',
    'trainable_encoder_blocks':[23],
}


class LastSSLFeatureGating(PartialSSLFeatureGating):
    def __init__(self, feature_extractor=None):
        super().__init__(feature_extractor)
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.encoder.layers[-1].requires_grad_(True)
