"""C0–C4: change WavLM fusion while retaining the full language-anchor backend."""

import torch
from torch import nn

from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss
from audio_deepfake_detection.lacf.precision import FP32LayerNorm


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
GROUPS = ((0, 4), (4, 8), (8, 13))  # Half-open representation indices, including R0.
PARAMETERS = dict(zip(CONDITIONS, (661479, 661479, 661492, 963498, 967338)))


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
