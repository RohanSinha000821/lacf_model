"""Six fixed controls; encoders and the original concept bank stay frozen."""

import torch
from torch import nn

from audio_deepfake_detection.lacf.model import (
    CONCEPTS, TEMPLATES, anchor_distribution, relation_classifier, unit_vectors,
)
from audio_deepfake_detection.lacf.precision import FP32LayerNorm
from audio_deepfake_detection.lacf_moe.model import LACFMoE


NAMES = {
    'A0': 'reference', 'A1': 'clap_embedding', 'A2': 'embedding_concat',
    'A3': 'wavlm_anchors', 'A4': 'uniform_layers',
    'A5': 'layer_and_fusion_moe', 'A6': 'global_fusion_routing',
}
FEATURE_INPUTS = {'A0': 'concat', 'A1': 'clap', 'A2': 'concat', 'A3': 'weighted',
                  'A4': 'concat', 'A5': 'concat', 'A6': 'concat'}
PARAMETERS = {'A0': 967338, 'A1': 35041, 'A2': 1032612, 'A3': 965556,
              'A4': 661479, 'A5': 981463, 'A6': 981288}
CORE = tuple(NAMES)
ADDITIONAL = {
    'B1': 'wavlm_embedding', 'B2': 'clap_anchors', 'B3': 'anchor_pair',
    'B4': 'agreement_difference', 'B5': 'global_layer_routing', 'B6': 'fusion_moe_only',
}


class FusionMoE(nn.Module):
    """Dense residual experts over the joint 35-D relation vector."""

    def __init__(self, *, global_routing=False):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(FP32LayerNorm(35), nn.Linear(35, 64), nn.GELU(), nn.Linear(64, 35))
            for _ in range(3)
        ])
        self.router = None if global_routing else nn.Sequential(FP32LayerNorm(35), nn.Linear(35, 3))
        self.global_logits = nn.Parameter(torch.zeros(3)) if global_routing else None
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

    def forward(self, relations):
        experts = torch.stack([relations + expert(relations) for expert in self.experts], dim=1)
        logits = (self.global_logits.expand(len(relations), -1) if self.router is None
                  else self.router(relations))
        with torch.autocast(relations.device.type, enabled=False):
            routing = logits.float().softmax(-1)
            return (experts.float() * routing[:, :, None]).sum(1), routing


class Ablation(nn.Module):
    def __init__(self, variant, prototypes):
        super().__init__()
        if variant not in CORE[1:]:
            raise ValueError('Only the fixed core controls A1–A6 can be fitted')
        self.variant = variant
        # Construct common weights/RNG exactly as the main reference, then retain
        # only applicable modules. New head/fusion initialization is isolated.
        reference = LACFMoE(prototypes)
        self.objective = reference.objective
        if variant != 'A1':
            self.adapter = reference.backend.adapter
        if variant not in ('A1', 'A4'):
            self.layer_fusion = reference.layer_fusion
        if variant in ('A3', 'A4', 'A5', 'A6'):
            self.register_buffer('prototypes', reference.backend.prototypes.clone())
        if variant in ('A4', 'A5', 'A6'):
            self.relations = reference.backend.fusion
        dimension = {'A1': 512, 'A2': 1024, 'A3': 8}.get(variant, 35)
        with torch.random.fork_rng(devices=[]):
            self.classifier = (reference.backend.classifier if dimension == 35
                               else relation_classifier(dimension))
            if dimension != 35:
                for index in range(4, len(self.classifier)):
                    self.classifier[index].load_state_dict(reference.backend.classifier[index].state_dict())
            if variant in ('A5', 'A6'):
                self.fusion_moe = FusionMoE(global_routing=variant == 'A6')
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count != PARAMETERS[variant]:
            raise ValueError(f'{variant} parameter inventory changed: {count}')

    def forward(self, batch):
        views = {}
        if self.variant != 'A1':
            if self.variant == 'A4':
                with torch.autocast(batch['wavlm'].device.type, enabled=False):
                    pooled = batch['wavlm'].float().mean(1)
            else:
                pooled, views['routing'] = self.layer_fusion(batch['wavlm'])
            views['v_w'] = unit_vectors(self.adapter(pooled))
        if self.variant != 'A3':
            views['v_c'] = unit_vectors(unit_vectors(batch['clap']))
        if self.variant in ('A3', 'A4', 'A5', 'A6'):
            views['p_w'] = anchor_distribution(views['v_w'], self.prototypes, 0.07)
            if self.variant != 'A3':
                views['p_c'] = anchor_distribution(views['v_c'], self.prototypes, 0.07)
        if self.variant == 'A1':
            features = views['v_c']
        elif self.variant == 'A2':
            features = torch.cat((views['v_w'], views['v_c']), -1)
        elif self.variant == 'A3':
            features = views['p_w']
        else:
            features = self.relations(views)
        if self.variant in ('A5', 'A6'):
            features, views['fusion_routing'] = self.fusion_moe(features)
        return {**views, 'features': features, 'logit': self.classifier(features).squeeze(-1)}

    def configuration(self):
        anchors = self.variant in ('A3', 'A4', 'A5', 'A6')
        return {
            'study': 'lacf_moe_ablations', 'variant': self.variant, 'name': NAMES[self.variant],
            'trainable_parameters': PARAMETERS[self.variant], 'foundation_encoders_loaded': False,
            'wavlm_enabled': self.variant != 'A1', 'clap_audio_enabled': self.variant != 'A3',
            'layer_fusion': ('disabled' if self.variant == 'A1' else 'uniform raw 13-layer mean'
                             if self.variant == 'A4' else 'original three-expert input-dependent MoE'),
            'fusion_moe': ('global learned softmax' if self.variant == 'A6' else 'input-dependent softmax'
                           if self.variant == 'A5' else None),
            'fusion_experts': '3 × residual LN35→64→GELU→35' if self.variant in ('A5', 'A6') else None,
            'adapter': [768, 512, 512] if self.variant != 'A1' else None,
            'classifier': [{'A1': 512, 'A2': 1024, 'A3': 8}.get(self.variant, 35), 64, 16, 1],
            'concepts': list(CONCEPTS) if anchors else None, 'templates': list(TEMPLATES) if anchors else None,
            'temperature': 0.07 if anchors else None, 'semantic_loss': False, 'bonafide_loss': False,
            'semantic_weight': 0, 'bonafide_weight': 0, 'objective': 'detection BCE only',
            'sensitive_dtype': 'float32', 'score_precision': 'reference BF16 autocast scalar logit, exported float32',
            'feature_inputs': FEATURE_INPUTS[self.variant],
        }


def build_model(variant, prototypes):
    if variant == 'A0':
        return LACFMoE(prototypes)
    return Ablation(variant, prototypes)
