"""TF0–TF3 frozen-cache controls; independent of primary LACF and SOTA code."""

import torch
from torch import nn

from audio_deepfake_detection.lacf.model import (
    CONCEPTS, TEMPLATES, ComponentConfig, LACFLoss, anchor_distribution,
    relation_classifier, unit_vectors, wavlm_adapter,
)
from audio_deepfake_detection.lacf.precision import FP32LayerNorm


CONDITIONS = ("TF0", "TF1", "TF2", "TF3")
PARAMETERS = {"TF0": 727777, "TF1": 4907233, "TF2": 4907745, "TF3": 5170401}


class AttentionBlock(nn.Module):
    """Pre-LN encoder block with explicit FP32 attention logits/probabilities."""
    def __init__(self):
        super().__init__()
        self.norm1 = FP32LayerNorm(512)
        self.qkv = nn.Linear(512, 1536)
        self.projection = nn.Linear(512, 512)
        self.norm2 = FP32LayerNorm(512)
        self.feedforward = nn.Sequential(nn.Linear(512, 1024), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(1024, 512))
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def probabilities(q, k):
        with torch.autocast(q.device.type, enabled=False):
            return (q.float() @ k.float().transpose(-2, -1) / 8.0).softmax(-1)

    def forward(self, tokens):
        batch, count, _ = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).reshape(batch, count, 3, 8, 64).permute(2, 0, 3, 1, 4)
        probabilities = self.dropout(self.probabilities(q, k))
        with torch.autocast(tokens.device.type, enabled=False):
            attended = (probabilities.float() @ v.float()).transpose(1, 2).reshape(batch, count, 512)
        tokens = tokens.float() + self.dropout(self.projection(attended))
        return tokens + self.dropout(self.feedforward(self.norm2(tokens)))


class Objective(nn.Module):
    """Reuse primary BCE numerics; optional semantic loss acts on fused CLS."""
    def __init__(self, semantic):
        super().__init__()
        self.loss = LACFLoss(ComponentConfig(semantic_loss=semantic, bonafide_loss=False,
                                            semantic_weight=0.25 if semantic else 0, bonafide_weight=0))

    def forward(self, output, labels):
        views = {"logit": output["logit"]}
        if self.loss.config.semantic_loss:
            # Shared numerical helper calls the supervised distribution p_w;
            # here its explicitly recorded scope is fused CLS, not WavLM alone.
            views["p_w"] = output["p_f"]
        return self.loss(views, labels)


class TokenFusion(nn.Module):
    def __init__(self, condition, prototypes=None):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("Expected TF0, TF1, TF2 or TF3")
        self.condition = condition
        self.uses_clap = condition != "TF1"
        self.adapter = wavlm_adapter()
        self.token_norm = FP32LayerNorm(512)
        self.classifier = relation_classifier(1024 if condition == "TF0" else 512)
        self.objective = Objective(condition == "TF3")
        # TF1/TF2/TF3 share initial adapter/head/block/CLS/layer weights and
        # the post-construction dropout RNG. TF3's projection is an extra arm.
        with torch.random.fork_rng(devices=[]):
            if condition != "TF0":
                self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
                self.cls = nn.Parameter(torch.empty(1, 1, 512))
                self.layer_ids = nn.Parameter(torch.empty(1, 13, 512))
                nn.init.normal_(self.cls, std=0.02)
                nn.init.normal_(self.layer_ids, std=0.02)
                identities = torch.empty(1, 2, 512).normal_(std=0.02)
                self.type_ids = nn.Parameter(identities[:, :2 if self.uses_clap else 1].clone())
            if condition == "TF3":
                if (prototypes is None or prototypes.shape != (8, 512)
                        or not torch.isfinite(prototypes).all()
                        or not torch.allclose(prototypes.float().norm(dim=-1), torch.ones_like(prototypes[:, 0]), atol=1e-5)):
                    raise ValueError("TF3 requires the eight fixed normalized original prototypes")
                self.register_buffer("prototypes", prototypes.detach().float().clone())
                self.semantic_projection = nn.Linear(512, 512)
        if self.configuration()["trainable_parameters"] != PARAMETERS[condition]:
            raise ValueError("Token-fusion parameter inventory changed")

    def tokens(self, batch):
        if batch["wavlm"].ndim != 3 or batch["wavlm"].shape[1:] != (13, 768):
            raise ValueError("Expected thirteen pooled WavLM vectors")
        wavlm = self.token_norm(self.adapter(batch["wavlm"]))
        clap = None
        if self.uses_clap:
            if batch["clap"].shape != (len(wavlm), 512):
                raise ValueError("Expected one CLAP projected vector per recording")
            clap = self.token_norm(unit_vectors(unit_vectors(batch["clap"]))).unsqueeze(1)
        if self.condition == "TF0":
            return torch.cat((wavlm.mean(1), clap[:, 0]), dim=-1)
        wavlm = wavlm + self.layer_ids + self.type_ids[:, :1]
        parts = [self.cls.expand(len(wavlm), -1, -1), wavlm]
        if clap is not None:
            parts.append(clap + self.type_ids[:, 1:2])
        return torch.cat(parts, dim=1)

    def forward(self, batch):
        features = self.tokens(batch)
        if self.condition != "TF0":
            for block in self.blocks:
                features = block(features)
            features = features[:, 0]
        output = {"features": features.float(), "logit": self.classifier(features).squeeze(-1)}
        if self.condition == "TF3":
            output["v_f"] = unit_vectors(self.semantic_projection(features))
            output["p_f"] = anchor_distribution(output["v_f"], self.prototypes, 0.07)
        return output

    def configuration(self):
        return {"condition": self.condition, "foundation_encoders_loaded": False,
                "wavlm": "13 cached raw validity-pooled 768-D representations",
                "clap_audio": self.uses_clap, "adapter_shared_across_layers": True,
                "adapter": [768, 512, 512], "adapter_dropout": 0.2,
                "input_token_normalization": "one shared FP32 LN512; no WavLM token L2",
                "clap_normalization": "two original FP32 L2 normalizations before token LN",
                "layer_and_type_ids": self.condition != "TF0", "identity_init_std": 0.02,
                "tokens_including_cls": None if self.condition == "TF0" else 15 if self.uses_clap else 14,
                "fusion": "mean projected WavLM + CLAP concat" if self.condition == "TF0" else
                          {"blocks": 2, "width": 512, "heads": 8, "feedforward": 1024,
                           "pre_norm": True, "activation": "GELU", "dropout": 0.1,
                           "attention_logits_softmax_value_aggregation": "float32",
                           "terminal_norm": False, "scope": "layers/encoder identity, not time"},
                "classifier": [1024 if self.condition == "TF0" else 512, 64, 16, 1],
                "classifier_order": "LN→Linear→GELU→Dropout(.2), twice, then Linear→1",
                "semantic_loss": self.condition == "TF3", "semantic_weight": 0.25 if self.condition == "TF3" else 0,
                "semantic_scope": "fused CLS" if self.condition == "TF3" else None,
                "semantic_projection": [512, 512] if self.condition == "TF3" else None,
                "concepts": list(CONCEPTS) if self.condition == "TF3" else None,
                "templates": list(TEMPLATES) if self.condition == "TF3" else None,
                "temperature": 0.07 if self.condition == "TF3" else None,
                "bonafide_js_loss": False, "text_enters_detection_head": False,
                "objective": "detection BCE + .25 fused semantic BCE" if self.condition == "TF3" else "detection BCE only",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "capacity_limit": "TF0 differs in capacity; TF1/2 differ by CLAP token/type vector; no novelty/target claim"}
