"""Owner-approved final-block F4 adaptation; same WavLM-WA forward and state keys."""
from pathlib import Path

from audio_deepfake_detection.sota.wavlm_wa import WavLMWA

PRETRAINED_REVISION = 'efa81aae7ff777e464159e0f877d54eac5b84f81'
PRETRAINED_PATH = Path('/home/ubuntu/.cache/huggingface/hub/models--microsoft--wavlm-base/snapshots') / PRETRAINED_REVISION


def freeze_except_last_block(model):
    """Disable unwanted frozen-prefix autograd without changing its train behavior."""
    if len(model.wavlm.encoder.layers) != 12:
        raise ValueError('Expected the twelve-block WavLM Base encoder')
    model.wavlm.requires_grad_(False)
    model.wavlm.encoder.layers[11].requires_grad_(True)
    # Transformers otherwise forces input gradients in the frozen convolution stack.
    model.wavlm.feature_extractor._requires_grad = False
    model.layer_weights.requires_grad_(True)
    model.classifier.requires_grad_(True)
    return model


class LastBlockWavLMWA(WavLMWA):
    def __init__(self):
        if not (PRETRAINED_PATH / 'config.json').is_file():
            raise FileNotFoundError('Pinned local WavLM Base cache is required; no download')
        super().__init__(model_name=str(PRETRAINED_PATH))
        freeze_except_last_block(self)

    def adaptation_configuration(self):
        encoder = sum(p.numel() for p in self.wavlm.parameters() if p.requires_grad)
        backend = self.layer_weights.numel() + sum(p.numel() for p in self.classifier.parameters())
        return {'fine_tuning': 'last_transformer_block_and_weighted_average_backend',
            'trainable_encoder_blocks': [11], 'encoder_blocks_total': 12,
            'pretrained_model': 'microsoft/wavlm-base', 'pretrained_revision': PRETRAINED_REVISION,
            'trainable_encoder_parameters': encoder, 'trainable_backend_parameters': backend,
            'trainable_parameters': encoder + backend,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'frozen_frontend_forced_input_gradients': False,
            'frozen_prefix_training_behavior': 'original dropout/layerdrop retained; SpecAugment remains off',
            'comparison_disclosure': 'F4 last-block fine-tuning; F1-F3 fully fine-tuned; same family, different adaptation scope'}
