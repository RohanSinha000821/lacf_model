"""Previously defined B1–B6 controls; original cache, prototypes and BCE only."""

import torch
from torch import nn

from audio_deepfake_detection.experiments.lacf_moe_ablations.model import FusionMoE
from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusion
from audio_deepfake_detection.lacf.model import CONCEPTS, TEMPLATES, RelationFeatures, anchor_distribution, relation_classifier, unit_vectors
from audio_deepfake_detection.lacf_moe.model import LACFMoE

NAMES = {'B1':'wavlm_embedding','B2':'clap_anchors','B3':'anchor_pair',
         'B4':'agreement_difference','B5':'global_layer_routing','B6':'fusion_moe_only'}
VARIANTS = tuple(NAMES)
FEATURE_INPUTS = {'B1':'weighted','B2':'clap',**{v:'concat' for v in VARIANTS[2:]}}
PARAMETERS = {'B1':998820,'B2':1777,'B3':966084,'B4':967140,'B5':963498,'B6':675604}
GROUPS = {'B3':('p_w','p_c'),'B4':('p_w','p_c','difference','agreement')}


class AdditionalAblation(nn.Module):
    def __init__(self,variant,prototypes):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Expected fixed additional B1–B6 control')
        reference=LACFMoE(prototypes)
        self.variant=variant;self.objective=reference.objective
        if variant!='B2':self.adapter=reference.backend.adapter
        if variant not in ('B2','B6'):
            self.layer_fusion=reference.layer_fusion
        if variant!='B1':self.register_buffer('prototypes',reference.backend.prototypes.clone())
        self.dimension={'B1':512,'B2':8,'B3':16,'B4':32}.get(variant,35)
        with torch.random.fork_rng(devices=[]):
            self.classifier=reference.backend.classifier if self.dimension==35 else relation_classifier(self.dimension)
            if self.dimension!=35:
                for i in range(4,len(self.classifier)):
                    self.classifier[i].load_state_dict(reference.backend.classifier[i].state_dict())
            if variant=='B6':self.fusion_moe=FusionMoE()
        if variant=='B5':
            # Match the original experts/initial uniform predictions; discard only its router.
            with torch.random.fork_rng(devices=[]):self.layer_fusion=LayerFusion('C3')
        if variant in ('B3','B4'):self.relations=RelationFeatures(GROUPS[variant])
        elif variant in ('B5','B6'):self.relations=reference.backend.fusion
        if sum(p.numel() for p in self.parameters())!=PARAMETERS[variant]:
            raise ValueError('Additional ablation parameter inventory changed')

    def forward(self,batch):
        views={}
        if self.variant!='B2':
            if self.variant=='B6':
                with torch.autocast(batch['wavlm'].device.type,enabled=False):
                    pooled=batch['wavlm'].float().mean(1)
            else:pooled,views['routing']=self.layer_fusion(batch['wavlm'])
            views['v_w']=unit_vectors(self.adapter(pooled))
        if self.variant!='B1':
            views['v_c']=unit_vectors(unit_vectors(batch['clap']))
            views['p_c']=anchor_distribution(views['v_c'],self.prototypes,.07)
            if self.variant!='B2':views['p_w']=anchor_distribution(views['v_w'],self.prototypes,.07)
        if self.variant=='B1':features=views['v_w']
        elif self.variant=='B2':features=views['p_c']
        else:features=self.relations(views)
        if self.variant=='B6':features,views['fusion_routing']=self.fusion_moe(features)
        return {**views,'features':features,'logit':self.classifier(features).squeeze(-1)}

    def configuration(self):
        anchors=self.variant!='B1'
        return {'study':'lacf_moe_ablations','variant':self.variant,'name':NAMES[self.variant],
            'trainable_parameters':PARAMETERS[self.variant],'foundation_encoders_loaded':False,
            'wavlm_enabled':self.variant!='B2','clap_audio_enabled':self.variant!='B1',
            'layer_fusion':('disabled' if self.variant=='B2' else 'uniform raw 13-layer mean' if self.variant=='B6'
                else 'same three residual experts; global learned softmax weights' if self.variant=='B5'
                else 'original three-expert input-dependent MoE'),
            'fusion_moe':'input-dependent residual35-D softmax' if self.variant=='B6' else None,
            'adapter':[768,512,512] if self.variant!='B2' else None,
            'classifier':[self.dimension,64,16,1],'dropout':.2,'classifier_normalization':'pre_linear_first_two_layers',
            'feature_groups':list(GROUPS[self.variant]) if self.variant in GROUPS else None,
            'concepts':list(CONCEPTS) if anchors else None,'templates':list(TEMPLATES) if anchors else None,
            'temperature':.07 if anchors else None,'semantic_loss':False,'bonafide_loss':False,
            'semantic_weight':0,'bonafide_weight':0,'objective':'detection BCE only',
            'feature_inputs':FEATURE_INPUTS[self.variant],'sensitive_dtype':'float32',
            'score_precision':'reference BF16 autocast scalar logit, exported float32',
            'common_initialization':'original adapter/expert weights and classifier tail; new blocks in isolated RNG'}


def build_model(variant,prototypes):
    return AdditionalAblation(variant,prototypes)
