"""Frozen WavLM Base+ cache with the trainable weighted-average linear backend."""
import torch
from torch import nn


class FrozenWavLMBasePlusWA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_weights=nn.Parameter(torch.zeros(13))
        self.classifier=nn.Linear(768,2)
        if sum(p.numel()for p in self.parameters())!=1551:
            raise ValueError('Weighted-average backend parameter count changed')

    def forward(self,batch):
        values=batch['wavlm']
        if values.ndim!=3 or values.shape[1:]!=(13,768):
            raise ValueError('Expected pooled WavLM Base+ layers [B,13,768]')
        with torch.autocast(values.device.type,enabled=False):
            weights=self.layer_weights.float().softmax(0)
            pooled=(values.float()*weights[None,:,None]).sum(1)
        logits=self.classifier(pooled)
        return {'logits':logits,'logit':logits[:,1]-logits[:,0],'layer_weights':weights}

    def configuration(self):
        return {'model':'Frozen WavLM-WA (Base+)','pretrained':'microsoft/wavlm-base-plus',
                'encoder_trainable':False,'foundation_encoders_loaded':False,'trainable_parameters':1551,
                'layer_mix':'13 trainable softmax weights, uniform initialization',
                'classifier':[768,2],'score':'spoof logit minus bona-fide logit',
                'pooling':'cached native validity-mask mean per layer; mix after temporal pooling',
                'disclosure':'Adapted frozen Base+ baseline; not the original Base/full-fine-tuning reproduction'}
