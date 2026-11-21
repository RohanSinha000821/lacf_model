"""Two capacity-matched embedding paths alongside unchanged MoE anchor relations."""
import torch
from torch import nn

from audio_deepfake_detection.lacf.model import anchor_distribution, relation_classifier, unit_vectors
from audio_deepfake_detection.lacf.precision import FP32LayerNorm
from audio_deepfake_detection.lacf_moe.model import LACFMoE

VARIANTS = ('clap_embedding_path', 'wavlm_embedding_control')
NAMES = {'clap_embedding_path':'relations + direct CLAP audio embedding',
         'wavlm_embedding_control':'relations + direct WavLM embedding (capacity control)'}
PARAMETERS = {v:1_005_418 for v in VARIANTS}
FEATURE_INPUTS = {v:'concat' for v in VARIANTS}
OBJECTIVE = 'detection BCE only'


class EmbeddingPathMoE(LACFMoE):
    def __init__(self, variant, prototypes):
        if variant not in VARIANTS:
            raise ValueError('Expected the fixed CLAP path or matched WavLM control')
        super().__init__(prototypes)
        self.variant = variant
        original_head = self.backend.classifier
        # Match all new weights and preserve the subsequent dropout RNG across arms.
        with torch.random.fork_rng(devices=[]):
            self.embedding_path = nn.Sequential(FP32LayerNorm(512), nn.Linear(512, 64), nn.GELU())
            self.backend.classifier = relation_classifier(99)
            for index in range(4, len(original_head)):
                self.backend.classifier[index].load_state_dict(original_head[index].state_dict())
        if sum(p.numel() for p in self.parameters() if p.requires_grad) != PARAMETERS[variant]:
            raise ValueError('Embedding-path parameter inventory changed')

    def forward(self, batch):
        pooled, routing = self.layer_fusion(batch['wavlm'])
        v_w = unit_vectors(self.backend.adapter(pooled))
        v_c = unit_vectors(unit_vectors(batch['clap']))
        views = {'v_w':v_w, 'v_c':v_c,
                 'p_w':anchor_distribution(v_w, self.backend.prototypes, .07),
                 'p_c':anchor_distribution(v_c, self.backend.prototypes, .07)}
        relations = self.backend.fusion(views)
        extra = self.embedding_path(v_c if self.variant == 'clap_embedding_path' else v_w)
        features = torch.cat((relations, extra), dim=-1)
        return {**views, 'routing':routing, 'relations':relations, 'embedding_features':extra,
                'features':features, 'logit':self.backend.classifier(features).squeeze(-1)}

    def configuration(self):
        config = super().configuration()
        config.pop('condition', None)
        config['backend']['classifier'] = [99, 64, 16, 1]
        config.update(study='lacf_moe_embedding_paths', variant=self.variant, name=NAMES[self.variant],
            trainable_parameters=PARAMETERS[self.variant], objective=OBJECTIVE,
            relation_dimension=35, embedding_source='normalized CLAP audio embedding' if self.variant == 'clap_embedding_path'
                else 'normalized adapted WavLM embedding',
            embedding_path='LN(512) -> Linear(512,64) -> GELU', embedding_dimension=64,
            concatenation=[35,64], classifier=[99,64,16,1],
            score_precision='FP32; autocast and TF32 disabled by guarded trainer',
            initialization='original MoE adapter/experts/router/head tail; bit-identical new path/head weights within seed',
            foundation_encoders_loaded=False, feature_inputs='concat')
        return config


def build_model(variant, prototypes):
    return EmbeddingPathMoE(variant, prototypes)
