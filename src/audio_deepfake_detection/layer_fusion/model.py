"""C0–C4: change WavLM fusion while retaining the full language-anchor backend."""

import torch
from torch import nn

from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.model import (
    CONCEPTS, TEMPLATES, ComponentConfig, LACFLoss, anchor_distribution, relation_classifier, unit_vectors,
)
from audio_deepfake_detection.lacf.precision import FP32LayerNorm


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
GROUPS = ((0, 4), (4, 8), (8, 13))  # Half-open representation indices, including R0.
PARAMETERS = dict(zip(CONDITIONS, (661479, 661479, 661492, 963498, 967338)))
READOUT_CONDITIONS = ("D1", "D2")
READOUT_PARAMETERS = {"D1": 659710, "D2": 692974}


class ResidualExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 64),
                                     nn.GELU(), nn.Linear(64, 768))

    def forward(self, inputs):
        return inputs + self.network(inputs)


class LayerFusion(nn.Module):
    def __init__(self, condition):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("Expected a frozen C0–C4 condition")
        self.condition = condition
        self.layer_weights = nn.Parameter(torch.zeros(13)) if condition == "C2" else None
        self.experts = nn.ModuleList([ResidualExpert() for _ in GROUPS]) if condition in ("C3", "C4") else None
        self.expert_weights = nn.Parameter(torch.zeros(3)) if condition == "C3" else None
        self.router = nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 3)) if condition == "C4" else None
        if self.router is not None:
            # Match C3's initial uniform routing and predictions. Experts still differ.
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

    def forward(self, layers):
        with torch.autocast(layers.device.type, enabled=False):
            layers = layers.float()
            if self.condition == "C0":
                return layers[:, -1], None
            if self.condition == "C1":
                return layers.mean(1), None
            if self.condition == "C2":
                return (layers * self.layer_weights.float().softmax(0)[None, :, None]).sum(1), None
            groups = [layers[:, start:end].mean(1) for start, end in GROUPS]
        experts = torch.stack([expert(value) for expert, value in zip(self.experts, groups)], dim=1)
        routing = self.routing_weights(layers)
        with torch.autocast(layers.device.type, enabled=False):
            return (experts.float() * routing[:, :, None]).sum(1), routing

    def routing_weights(self, layers):
        if self.condition not in ("C3", "C4"):
            raise ValueError("Only expert conditions have routing weights")
        if self.router is None:
            logits = self.expert_weights.expand(len(layers), -1)
        else:
            with torch.autocast(layers.device.type, enabled=False):
                average = layers.float().mean(1)
            logits = self.router(average)
        with torch.autocast(layers.device.type, enabled=False):
            return logits.float().softmax(-1)


class LayerFusionDetector(nn.Module):
    def __init__(self, condition, prototypes):
        super().__init__()
        # Construct the common backend first, independent of fusion capacity.
        self.backend = RelationAblation("A9", prototypes)
        self.backend.config = ComponentConfig(semantic_loss=False, bonafide_loss=False,
                                              semantic_weight=0, bonafide_weight=0)
        self.backend.objective = LACFLoss(self.backend.config)
        # Fusion initialization must not alter the subsequent common dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.layer_fusion = LayerFusion(condition)
        self.condition = condition
        if self.configuration()["trainable_parameters"] != PARAMETERS[condition]:
            raise ValueError("Layer-fusion parameter inventory changed")

    @property
    def objective(self):
        return self.backend.objective

    @property
    def layer_weights(self):
        return self.layer_fusion.layer_weights

    def forward(self, batch):
        pooled, routing = self.layer_fusion(batch["wavlm"])
        output = self.backend({"wavlm": pooled, "clap": batch["clap"]})
        if routing is not None:
            output["routing"] = routing
        return output

    def configuration(self):
        backend = self.backend.configuration()
        del backend["variant"]
        backend["wavlm_features"] = "13 cached raw validity-mask-pooled representations; chosen fusion before adapter"
        return {"condition": self.condition, "backend": backend,
                "fusion": {"C0": "last representation", "C1": "uniform mean of 13 representations",
                           "C2": "13 global softmax logits, initialized uniformly",
                           "C3": "three residual experts with three global softmax logits",
                           "C4": "same experts with recording-dependent softmax router"}[self.condition],
                "expert_groups_half_open": [list(group) for group in GROUPS] if self.condition in ("C3", "C4") else None,
                "expert": "x + Linear(64,768)(GELU(Linear(768,64)(LN(x)))); no extra dropout"
                          if self.condition in ("C3", "C4") else None,
                "router": "LN(mean of 13 raw layers) → Linear(768,3) → softmax; final linear zero-initialized"
                          if self.condition == "C4" else None,
                "common_backend_initialization": "identical within seed; isolated fusion initialization RNG",
                "foundation_encoders_loaded": False, "objective": "detection BCE only",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


class ReadoutDetector(nn.Module):
    """D1 anchor-only / D2 direct readouts of the same normalized C2 adapter.

    A fresh reference backend preserves C2's adapter initialization and RNG
    consumption. Its unused classifier is discarded, never fitted or retained.
    Only the differently sized first classifier block gets a new initialization;
    the 64→16→1 tail is copied from the fresh reference in both controls.
    """
    def __init__(self, condition, prototypes):
        super().__init__()
        if condition not in READOUT_CONDITIONS:
            raise ValueError("Expected D1 or D2")
        reference = RelationAblation("A9", prototypes)
        self.adapter = reference.adapter
        self.condition = condition
        self.input_dim = 8 if condition == "D1" else 512
        if condition == "D1":
            self.register_buffer("prototypes", reference.prototypes.clone())
        with torch.random.fork_rng(devices=[]):
            self.classifier = relation_classifier(self.input_dim)
            self.layer_fusion = LayerFusion("C2")
        for index in range(4, len(self.classifier)):
            self.classifier[index].load_state_dict(reference.classifier[index].state_dict())
        self.objective = LACFLoss(ComponentConfig(semantic_loss=False, bonafide_loss=False,
                                                 semantic_weight=0, bonafide_weight=0))
        if self.configuration()["trainable_parameters"] != READOUT_PARAMETERS[condition]:
            raise ValueError("Readout parameter inventory changed")

    @property
    def layer_weights(self):
        return self.layer_fusion.layer_weights

    def forward(self, batch):
        pooled, _ = self.layer_fusion(batch["wavlm"])
        adapted = unit_vectors(self.adapter(pooled))
        features = (anchor_distribution(adapted, self.prototypes, 0.07)
                    if self.condition == "D1" else adapted)
        return {"v_w": adapted, "features": features,
                "logit": self.classifier(features).squeeze(-1)}

    def configuration(self):
        return {"condition": self.condition, "comparison": "multi_layer_readouts",
                "fusion": "13 global softmax logits, initialized uniformly; fitted separately in each arm",
                "wavlm_features": "13 cached raw validity-mask-pooled representations; no extra standardization",
                "adapter": [768, 512, 512], "embedding_normalization": "FP32 L2 after adapter",
                "classifier": [self.input_dim, 64, 16, 1], "dropout": 0.2,
                "classifier_normalization": "pre_linear_first_two_layers",
                "readout": "p_w only" if self.condition == "D1" else "normalized WavLM adapter embedding",
                "clap_audio": False, "anchor_probabilities": self.condition == "D1",
                "concepts": list(CONCEPTS) if self.condition == "D1" else None,
                "templates": list(TEMPLATES) if self.condition == "D1" else None,
                "temperature": 0.07 if self.condition == "D1" else None,
                "objective": "detection BCE only", "semantic_loss": False, "bonafide_loss": False,
                "semantic_weight": 0, "bonafide_weight": 0,
                "common_initialization": "C2 adapter, zero layer logits, 64→16→1 classifier tail and dropout RNG",
                "comparison_limitations": "first classifier block width/capacity differs; learned mixtures adapt separately",
                "foundation_encoders_loaded": False, "sensitive_dtype": "float32",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}
