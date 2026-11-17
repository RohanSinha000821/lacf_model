"""C1–C3: original three-expert LACF-MoE, changing auxiliary losses only."""

from dataclasses import replace

from audio_deepfake_detection.lacf.model import LACFLoss
from audio_deepfake_detection.lacf_moe.model import LACFMoE, TRAINABLE_PARAMETERS

NAMES = {'C1': 'bce_semantic', 'C2': 'bce_bonafide_js', 'C3': 'bce_semantic_bonafide_js'}
VARIANTS = tuple(NAMES)
WEIGHTS = {'C1': (.25, 0.), 'C2': (0., .10), 'C3': (.25, .10)}
OBJECTIVES = {'C1': 'detection BCE + 0.25 semantic BCE',
              'C2': 'detection BCE + 0.10 bona-fide JS',
              'C3': 'detection BCE + 0.25 semantic BCE + 0.10 bona-fide JS'}
PARAMETERS = {v: TRAINABLE_PARAMETERS for v in VARIANTS}
FEATURE_INPUTS = {v: 'concat' for v in VARIANTS}


class LossAblation(LACFMoE):
    def __init__(self, variant, prototypes):
        if variant not in VARIANTS:
            raise ValueError('Expected fixed loss control C1, C2 or C3')
        super().__init__(prototypes)
        self.variant = variant
        semantic, bonafide = WEIGHTS[variant]
        self.backend.config = replace(self.backend.config, semantic_loss=semantic != 0,
            bonafide_loss=bonafide != 0, semantic_weight=semantic, bonafide_weight=bonafide)
        self.backend.objective = LACFLoss(self.backend.config)

    def configuration(self):
        result = super().configuration()
        # The parent uses historical diagnostic condition C4 for this architecture;
        # current C1–C3 IDs describe losses, not those old layer-fusion conditions.
        result.pop('condition')
        result.update(study='lacf_moe_ablations', variant=self.variant, name=NAMES[self.variant],
            objective=OBJECTIVES[self.variant], architecture='unchanged main three-expert LACF-MoE',
            semantic_definition='BCE(sum of p_W over original spoof concepts indices3:8, binary label)',
            bonafide_definition='mean JS(p_W,p_C) over bona-fide samples only; zero for all-spoof batch',
            loss_dtype='float32', shared_initialization='bit-identical to fresh BCE-only reference within seed')
        return result


def build_model(variant, prototypes):
    return LossAblation(variant, prototypes)
