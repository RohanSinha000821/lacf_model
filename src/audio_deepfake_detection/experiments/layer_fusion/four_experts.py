"""Four-expert control of the existing frozen LACF Layer-MoE."""

import torch
from torch import nn

from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusion, LayerFusionDetector, ResidualExpert
from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss
from audio_deepfake_detection.lacf.precision import FP32LayerNorm


GROUPS = ((0, 4), (4, 7), (7, 10), (10, 13))
TRAINABLE_PARAMETERS = 1_068_779


class FourExpertFusion(LayerFusion):
    def __init__(self):
        nn.Module.__init__(self)
        self.condition = 'C4'
        self.layer_weights = self.expert_weights = None
        self.experts = nn.ModuleList([ResidualExpert() for _ in GROUPS])
        self.router = nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 4))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, layers):
        if layers.ndim != 3 or layers.shape[1:] != (13, 768):
            raise ValueError('Expected thirteen pooled 768-D WavLM representations')
        with torch.autocast(layers.device.type, enabled=False):
            layers = layers.float()
            groups = [layers[:, start:end].mean(1) for start, end in GROUPS]
        experts = torch.stack([expert(value) for expert, value in zip(self.experts, groups)], dim=1)
        routing = self.routing_weights(layers)
        with torch.autocast(layers.device.type, enabled=False):
            return (experts.float() * routing[:, :, None]).sum(1), routing


class FourExpertDetector(LayerFusionDetector):
    def __init__(self, prototypes):
        nn.Module.__init__(self)
        # Same backend construction/RNG consumption as the completed three-expert model.
        self.backend = RelationAblation('A9', prototypes)
        self.backend.config = ComponentConfig(semantic_loss=False, bonafide_loss=False,
                                              semantic_weight=0, bonafide_weight=0)
        self.backend.objective = LACFLoss(self.backend.config)
        with torch.random.fork_rng(devices=[]):
            self.layer_fusion = FourExpertFusion()
        self.condition = 'C4'
        if self.configuration()['trainable_parameters'] != TRAINABLE_PARAMETERS:
            raise ValueError('Four-expert parameter inventory changed')

    def configuration(self):
        result = super().configuration()
        result.update(condition='four_experts', model_name='LACF Layer-MoE (4 experts)',
                      fusion='four residual experts with recording-dependent softmax routing',
                      expert_count=4, expert_groups_half_open=[list(g) for g in GROUPS],
                      router='LN(mean of 13 raw layers) → Linear(768,4) → softmax; zero-initialized linear',
                      comparison_limits='Expert count, group boundaries and capacity differ from the three-expert reference')
        return result
