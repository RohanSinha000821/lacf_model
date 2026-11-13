"""Main three-expert LACF-MoE, numerically identical to the completed reference."""

import torch
from torch import nn

from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss
from audio_deepfake_detection.lacf.precision import FP32LayerNorm

GROUPS = ((0, 4), (4, 8), (8, 13))
TRAINABLE_PARAMETERS = 967_338


class ResidualExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 64), nn.GELU(), nn.Linear(64, 768))

    def forward(self, values):
        return values + self.network(values)


class LayerMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([ResidualExpert() for _ in GROUPS])
        self.router = nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 3))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def routing_weights(self, layers):
        with torch.autocast(layers.device.type, enabled=False):
            average = layers.float().mean(1)
        logits = self.router(average)
        with torch.autocast(layers.device.type, enabled=False):
            return logits.float().softmax(-1)

    def forward(self, layers):
        if layers.ndim != 3 or layers.shape[1:] != (13, 768):
            raise ValueError('Expected thirteen cached pooled 768-D WavLM representations')
        with torch.autocast(layers.device.type, enabled=False):
            layers = layers.float()
            groups = [layers[:, start:end].mean(1) for start, end in GROUPS]
        experts = torch.stack([expert(value) for expert, value in zip(self.experts, groups)], dim=1)
        routing = self.routing_weights(layers)
        with torch.autocast(layers.device.type, enabled=False):
            return (experts.float() * routing[:, :, None]).sum(1), routing


class LACFMoE(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        # Reuse the prescribed full relation backend, with detection BCE only.
        self.backend = RelationAblation('A9', prototypes)
        self.backend.config = ComponentConfig(semantic_loss=False, bonafide_loss=False,
                                              semantic_weight=0, bonafide_weight=0)
        self.backend.objective = LACFLoss(self.backend.config)
        with torch.random.fork_rng(devices=[]):
            self.layer_fusion = LayerMoE()
        if sum(p.numel() for p in self.parameters()) != TRAINABLE_PARAMETERS:
            raise ValueError('LACF-MoE parameter inventory changed')

    @property
    def objective(self):
        return self.backend.objective

    def forward(self, batch):
        pooled, routing = self.layer_fusion(batch['wavlm'])
        return {**self.backend({'wavlm': pooled, 'clap': batch['clap']}), 'routing': routing}

    def configuration(self):
        # Keep the original model metadata/state keys for exact historical F1 reuse.
        backend = self.backend.configuration()
        del backend['variant']
        backend['wavlm_features'] = '13 cached raw validity-mask-pooled representations; chosen fusion before adapter'
        return {'condition': 'C4', 'backend': backend,
                'fusion': 'same experts with recording-dependent softmax router',
                'expert_groups_half_open': [list(group) for group in GROUPS],
                'expert': 'x + Linear(64,768)(GELU(Linear(768,64)(LN(x)))); no extra dropout',
                'router': 'LN(mean of 13 raw layers) → Linear(768,3) → softmax; final linear zero-initialized',
                'common_backend_initialization': 'identical within seed; isolated fusion initialization RNG',
                'foundation_encoders_loaded': False, 'objective': 'detection BCE only',
                'trainable_parameters': TRAINABLE_PARAMETERS}
