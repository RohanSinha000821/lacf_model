"""The same two embedding paths, with the historical BF16 execution contract."""
import torch
from .model import EmbeddingPathMoE, VARIANTS, NAMES, PARAMETERS, FEATURE_INPUTS

PRECISION = 'BF16 mixed; sensitive FP32; reference BF16-rounded logits'


class BF16EmbeddingPathMoE(EmbeddingPathMoE):
    def forward(self, batch):
        device = batch['wavlm'].device.type
        if not torch.is_autocast_enabled(device):
            raise ValueError('Historical BF16 autocast required; no precision fallback')
        result = super().forward(batch)
        if result['logit'].dtype != torch.bfloat16:
            raise TypeError('Final classifier must compute BF16 logits before float32 export')
        return result

    def configuration(self):
        result = super().configuration()
        result['score_precision'] = PRECISION
        result['execution'] = 'historical BF16 autocast with unchanged FP32-sensitive operations'
        return result


def build_model(variant, prototypes):
    return BF16EmbeddingPathMoE(variant, prototypes)
