"""The main plan's A4–A9 frozen-feature relation and loss controls."""

from dataclasses import asdict

import torch
from torch import nn

from audio_deepfake_detection.lacf.model import (
    CLAP_NAME, CONCEPTS, TEMPLATES, WAVLM_NAME, ComponentConfig, LACFLoss,
    RelationFeatures, anchor_distribution, relation_classifier, unit_vectors, wavlm_adapter,
)


VARIANTS = {
    "A4": ComponentConfig(feature_groups=("p_c",), semantic_loss=False, bonafide_loss=False,
                          semantic_weight=0, bonafide_weight=0),
    "A5": ComponentConfig(feature_groups=("p_w",), bonafide_loss=False, bonafide_weight=0),
    "A6": ComponentConfig(feature_groups=("p_w", "p_c"), bonafide_loss=False, bonafide_weight=0),
    "A7": ComponentConfig(feature_groups=("p_w", "p_c", "difference", "agreement"),
                          bonafide_loss=False, bonafide_weight=0),
    "A8": ComponentConfig(bonafide_loss=False, bonafide_weight=0),
    "A9": ComponentConfig(),
    "A9_no_sem": ComponentConfig(semantic_loss=False, semantic_weight=0),
}
ALIASES = {"A9_no_bf": "A8"}


def feature_inputs(variant):
    config = VARIANTS[variant]
    return "concat" if config.needs_wavlm and config.needs_clap_audio else "last" if config.needs_wavlm else "clap"


class RelationAblation(nn.Module):
    """Train the prescribed backend from cached encoder outputs, never encoders."""
    def __init__(self, variant, prototypes):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown or duplicate ablation; A9_no_bf reuses A8")
        prototypes = prototypes.detach().float().cpu()
        if (prototypes.shape != (8, 512) or not torch.isfinite(prototypes).all()
                or not torch.allclose(prototypes.norm(dim=-1), torch.ones(8), atol=1e-6, rtol=1e-6)):
            raise ValueError("Expected the unchanged normalized primary text prototypes")
        self.variant, self.config = variant, VARIANTS[variant]
        # Preserve recorded buffer bytes; do not normalize the frozen prototypes again.
        self.register_buffer("prototypes", prototypes.clone())
        self.adapter = wavlm_adapter() if self.config.needs_wavlm else None
        self.fusion = RelationFeatures(self.config.feature_groups)
        self.classifier = relation_classifier(self.fusion.output_dim)
        self.objective = LACFLoss(self.config)

    def forward(self, batch):
        views = {}
        if self.config.needs_wavlm:
            views["z_w"] = batch["wavlm"]
            views["v_w"] = unit_vectors(self.adapter(views["z_w"]))
            views["p_w"] = anchor_distribution(views["v_w"], self.prototypes, self.config.temperature)
        if self.config.needs_clap_audio:
            # Cache stores the raw projection. Reproduce CLAP API normalization
            # followed by LACF's normalization, both in FP32, as in the primary.
            views["v_c"] = unit_vectors(unit_vectors(batch["clap"]))
            views["p_c"] = anchor_distribution(views["v_c"], self.prototypes, self.config.temperature)
        features = self.fusion(views)
        return {**views, "features": features, "logit": self.classifier(features).squeeze(-1)}

    def configuration(self):
        return {"variant": self.variant, **asdict(self.config), "fusion": self.fusion.configuration(),
                "adapter": [768, 512, 512] if self.adapter is not None else None,
                "classifier": [self.fusion.output_dim, 64, 16, 1], "dropout": 0.2,
                "classifier_normalization": "pre_linear_first_two_layers",
                "wavlm_name": WAVLM_NAME, "clap_name": CLAP_NAME,
                "concepts": list(CONCEPTS), "templates": list(TEMPLATES),
                "wavlm_features": "raw mask-pooled last layer; no learned layer mixing",
                "clap_features": "raw cached projection, normalized as CLAP API then primary LACF",
                "foundation_encoders_loaded": False, "sensitive_dtype": "float32",
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}
