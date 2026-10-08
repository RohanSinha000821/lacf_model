"""Frozen LACF components from the owner-designated implementation plan.

The pipeline supplement specifies pre-linear LayerNorm in the relation head.
Feature groups and loss switches prepare the prescribed later ablations; no
ablation is selected automatically and no optional encoder fine-tuning exists.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from audio_deepfake_detection.lacf.precision import (
    FP32LayerNorm, fp32_projection_output, keep_normalization_fp32,
)

WAVLM_NAME = "microsoft/wavlm-base-plus"
CLAP_NAME = "laion/clap-htsat-unfused"
CONCEPTS = (
    "natural human speech", "authentic human voice recording", "naturally produced human voice",
    "synthetic speech", "text-to-speech generated voice", "voice-converted speech",
    "AI-cloned human voice", "neural-vocoder generated speech",
)
TEMPLATES = ("{concept}", "a recording of {concept}", "this audio contains {concept}")
FEATURE_DIMS = {"p_w": 8, "p_c": 8, "difference": 8, "agreement": 8,
                "js": 1, "entropy_w": 1, "entropy_c": 1}
PRIMARY_TRAINABLE_PARAMETERS = 661_479


@dataclass(frozen=True)
class ComponentConfig:
    feature_groups: tuple[str, ...] = tuple(FEATURE_DIMS)
    semantic_loss: bool = True
    bonafide_loss: bool = True
    temperature: float = 0.07
    semantic_weight: float = 0.25
    bonafide_weight: float = 0.10

    def __post_init__(self):
        if (not self.feature_groups or len(set(self.feature_groups)) != len(self.feature_groups)
                or any(group not in FEATURE_DIMS for group in self.feature_groups)):
            raise ValueError("Feature groups must be nonempty, unique, prescribed relation groups")
        if not 0 < self.temperature < float("inf"):
            raise ValueError("Temperature must be finite and positive")
        if any(not 0 <= weight < float("inf") for weight in (self.semantic_weight, self.bonafide_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")

    @property
    def needs_wavlm(self):
        return self.semantic_loss or self.bonafide_loss or bool(set(self.feature_groups) - {"p_c", "entropy_c"})

    @property
    def needs_clap_audio(self):
        return self.bonafide_loss or bool(set(self.feature_groups) - {"p_w", "entropy_w"})


def unit_vectors(values):
    with torch.autocast(values.device.type, enabled=False):
        values = values.float()
        if values.ndim < 2 or not torch.isfinite(values).all() or (values.norm(dim=-1) == 0).any():
            raise ValueError("Embedding vectors must be finite and nonzero")
        return F.normalize(values, dim=-1)


def anchor_distribution(embedding, prototypes, temperature):
    with torch.autocast(embedding.device.type, enabled=False):
        return (embedding.float() @ prototypes.float().T / temperature).softmax(-1)


def text_prototypes(text_features):
    """Normalize each of 24 projected features, then average and normalize per concept."""
    if text_features.shape != (24, 512):
        raise ValueError("Expected 24 projected 512-dimensional CLAP text features")
    return unit_vectors(unit_vectors(text_features).reshape(8, 3, 512).mean(dim=1))


def entropy(probabilities):
    # Natural logarithms, as in the supplement. Clamp only inside log to handle 0*log(0).
    with torch.autocast(probabilities.device.type, enabled=False):
        probabilities = probabilities.float()
        return -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)


def js_divergence(p_w, p_c):
    with torch.autocast(p_w.device.type, enabled=False):
        p_w, p_c = p_w.float(), p_c.float()
        midpoint = (p_w + p_c) / 2
        tiny = torch.finfo(torch.float32).tiny
        log_midpoint = midpoint.clamp_min(tiny).log()
        return 0.5 * ((p_w * (p_w.clamp_min(tiny).log() - log_midpoint)).sum(-1)
                      + (p_c * (p_c.clamp_min(tiny).log() - log_midpoint)).sum(-1))


class RelationFeatures(nn.Module):
    def __init__(self, groups=tuple(FEATURE_DIMS)):
        super().__init__()
        self.groups = ComponentConfig(feature_groups=tuple(groups)).feature_groups
        self.output_dim = sum(FEATURE_DIMS[group] for group in self.groups)

    def configuration(self):
        return {"kind": "anchor_relations", "groups": list(self.groups), "output_dim": self.output_dim}

    def forward(self, views):
        p_w, p_c = views.get("p_w"), views.get("p_c")
        p_w = p_w.float() if p_w is not None else None
        p_c = p_c.float() if p_c is not None else None
        blocks = {}
        if p_w is not None:
            blocks.update(p_w=p_w, entropy_w=entropy(p_w).unsqueeze(-1))
        if p_c is not None:
            blocks.update(p_c=p_c, entropy_c=entropy(p_c).unsqueeze(-1))
        if p_w is not None and p_c is not None:
            blocks.update(difference=(p_w - p_c).abs(), agreement=p_w * p_c,
                          js=js_divergence(p_w, p_c).unsqueeze(-1))
        if any(group not in blocks for group in self.groups):
            raise ValueError("A configured relation feature requires a missing view")
        return torch.cat([blocks[group] for group in self.groups], dim=-1)


class LACFLoss(nn.Module):
    def __init__(self, config=ComponentConfig()):
        super().__init__()
        self.config = config

    def forward(self, output, labels):
        with torch.autocast(output["logit"].device.type, enabled=False):
            return self.fp32_loss(output, labels)

    def fp32_loss(self, output, labels):
        logit, labels = output["logit"].float(), labels.float()
        if labels.shape != output["logit"].shape or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Expected one binary label per spoof logit")
        detection = F.binary_cross_entropy_with_logits(logit, labels)
        zero = detection * 0
        semantic, bonafide = zero, zero
        if self.config.semantic_loss:
            semantic = F.binary_cross_entropy(output["p_w"].float()[:, 3:].sum(-1).clamp(0, 1), labels)
        if self.config.bonafide_loss:
            divergence = js_divergence(output["p_w"], output["p_c"])
            bonafide = divergence[labels == 0].mean() if (labels == 0).any() else zero
        total = detection + self.config.semantic_weight * semantic + self.config.bonafide_weight * bonafide
        return {"total": total, "detection": detection, "semantic": semantic, "bonafide": bonafide}


def relation_classifier(input_dim):
    return nn.Sequential(FP32LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(0.2),
                         FP32LayerNorm(64), nn.Linear(64, 16), nn.GELU(), nn.Dropout(0.2), nn.Linear(16, 1))


class LACF(nn.Module):
    """Injected pretrained encoders and fixed prototypes; construction never downloads.

    A later fusion replacement implements forward(views), output_dim and
    configuration(). Views include z_w, v_w, v_c and applicable anchor probabilities.
    Its declared output dimension constructs the same head without changing the loop.
    """

    def __init__(self, wavlm, clap, prototypes, *, config=ComponentConfig(), fusion=None):
        super().__init__()
        self.config = config
        self.wavlm = wavlm if config.needs_wavlm else None
        self.clap = clap if config.needs_clap_audio else None
        if config.needs_wavlm and (wavlm is None or wavlm.config.hidden_size != 768):
            raise ValueError("WavLM Base+ must expose 768-dimensional states")
        if config.needs_clap_audio and (clap is None or clap.config.projection_dim != 512):
            raise ValueError("CLAP must expose projected 512-dimensional audio features")
        if prototypes.shape != (8, 512):
            raise ValueError("Expected eight 512-dimensional fixed prototypes")
        self.register_buffer("prototypes", unit_vectors(prototypes.detach()).clone())
        self.adapter = (nn.Sequential(FP32LayerNorm(768), nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(512, 512)) if config.needs_wavlm else None)
        self.fusion = fusion if fusion is not None else RelationFeatures(config.feature_groups)
        if not isinstance(self.fusion.output_dim, int) or self.fusion.output_dim < 1:
            raise ValueError("Fusion must declare a positive output_dim")
        self.classifier = relation_classifier(self.fusion.output_dim)
        self.objective = LACFLoss(config)
        for encoder in (self.wavlm, self.clap):
            if encoder is not None:
                keep_normalization_fp32(encoder)
                encoder.requires_grad_(False)
                encoder.eval()
        if self.clap is not None and hasattr(self.clap, "audio_projection"):
            self.clap.audio_projection.register_forward_hook(fp32_projection_output)

    def train(self, mode=True):
        super().train(mode)
        # Frozen means no parameter updates AND no encoder dropout/SpecAugment/BN updates.
        for encoder in (self.wavlm, self.clap):
            if encoder is not None:
                encoder.eval()
        return self

    def configuration(self):
        return {**asdict(self.config), "fusion": self.fusion.configuration(), "stage": "frozen",
                "wavlm_name": WAVLM_NAME, "clap_name": CLAP_NAME,
                "concepts": list(CONCEPTS), "templates": list(TEMPLATES),
                "wavlm_enabled": self.wavlm is not None, "clap_audio_enabled": self.clap is not None,
                "prototype_policy": "normalize_each_template_then_mean_then_normalize",
                "adapter": [768, 512, 512] if self.adapter is not None else None,
                "dropout": 0.2, "classifier_normalization": "pre_linear_first_two_layers",
                "classifier": [self.fusion.output_dim, 64, 16, 1],
                "sensitive_calculations_dtype": "float32"}

    def forward(self, batch):
        views = {}
        if self.wavlm is not None:
            with torch.no_grad():
                mask = batch["wavlm"]["attention_mask"]
                if mask.ndim != 2 or not mask.bool().any(dim=1).all():
                    raise ValueError("Expected a nonempty validity mask for every WavLM input")
                hidden = self.wavlm(**batch["wavlm"], return_dict=True).last_hidden_state
                frame_mask = self.wavlm._get_feature_vector_attention_mask(hidden.shape[1], mask)
                if not frame_mask.any(dim=1).all():
                    raise ValueError("Audio is too short for a valid WavLM feature frame")
                with torch.autocast(hidden.device.type, enabled=False):
                    pooled = (hidden.float() * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1, keepdim=True)
            views["z_w"] = pooled
            views["v_w"] = unit_vectors(self.adapter(pooled))
            views["p_w"] = anchor_distribution(views["v_w"], self.prototypes, self.config.temperature)
        if self.clap is not None:
            with torch.no_grad():
                views["v_c"] = unit_vectors(self.clap.get_audio_features(**batch["clap"]))
                views["p_c"] = anchor_distribution(views["v_c"], self.prototypes, self.config.temperature)
        features = self.fusion(views)
        if features.ndim != 2 or features.shape[1] != self.fusion.output_dim:
            raise ValueError("Fusion output disagrees with its declared feature dimension")
        return {**views, "features": features, "logit": self.classifier(features).squeeze(-1)}

    @staticmethod
    def spoof_score(output):
        return output["logit"]
