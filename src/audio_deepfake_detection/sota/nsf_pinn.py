"""Frozen WavLM Base+ NSF-PINN detector and differentiable physics objectives.

Equations 13/19/23/27/30 govern the objective. The paper's algorithm differs;
all resolutions and unreported widths/weights are documented study choices.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NSFPINNConfig:
    width: int = 768
    hidden: int = 256
    embedding: int = 128
    classifier_hidden: int = 256
    dropout: float = 0.2
    reconstruction_weight: float = 0.1
    smoothness_weight: float = 0.1
    orthogonality_weight: float = 0.1
    warmup_epochs: int = 2


class NSFPINNDetector(nn.Module):
    def __init__(self, cfg=NSFPINNConfig()):
        super().__init__()
        if min(cfg.width, cfg.hidden, cfg.embedding, cfg.classifier_hidden, cfg.warmup_epochs) < 1:
            raise ValueError("Invalid NSF-PINN dimensions/warmup")
        if not 0 <= cfg.dropout < 1 or min(cfg.reconstruction_weight, cfg.smoothness_weight,
                                         cfg.orthogonality_weight) < 0:
            raise ValueError("Invalid NSF-PINN dropout/loss weights")
        self.cfg = cfg
        self.ssl_projection = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.GELU(),
                                            nn.Linear(cfg.width, cfg.embedding))
        self.temporal = nn.Sequential(nn.Linear(cfg.width, cfg.hidden), nn.GELU(),
                                       nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.source = nn.Linear(cfg.hidden, cfg.embedding)
        self.filter = nn.Linear(cfg.hidden, cfg.embedding)
        self.reconstruction = nn.Sequential(nn.Linear(2 * cfg.embedding, cfg.hidden), nn.GELU(),
                                             nn.Linear(cfg.hidden, cfg.width))
        self.classifier = nn.Sequential(nn.Linear(3 * cfg.embedding, cfg.classifier_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.classifier_hidden, 2))

    def forward(self, frames, *, return_physics=False):
        if frames.ndim != 3 or frames.shape[-1] != self.cfg.width or frames.shape[1] < 1:
            raise ValueError("Expected [batch,time,WavLM width] frame-level features")
        h = frames.mean(dim=1)
        z = self.temporal(frames)
        source, filtr = self.source(z), self.filter(z)
        s, f = source.mean(dim=1), filtr.mean(dim=1)
        logits = self.classifier(torch.cat((self.ssl_projection(h), s, f), dim=1))
        if return_physics:
            return {"logits": logits, "source": source, "filter": filtr, "s": s, "f": f,
                    "h": h, "reconstructed": self.reconstruction(torch.cat((s, f), dim=1))}
        return logits

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]


def physics_loss(outputs, labels, cfg, *, epoch):
    """FP32 reductions; temporal differences never cross utterance boundaries."""
    if epoch < 1:
        raise ValueError("Epoch numbering starts at one")
    o = {key: value.float() for key, value in outputs.items()}
    classification = F.cross_entropy(o["logits"], labels)
    reconstruction = F.mse_loss(o["reconstructed"], o["h"])
    if o["source"].shape[1] > 1:
        smoothness = (o["source"].diff(dim=1).square().mean()
                      + o["filter"].diff(dim=1).square().mean())
    else:
        smoothness = (o["source"].sum() + o["filter"].sum()) * 0
    cosine = (F.normalize(o["s"], dim=-1) * F.normalize(o["f"], dim=-1)).sum(dim=-1)
    orthogonality = cosine.square().mean()
    alpha = min(1.0, epoch / cfg.warmup_epochs)
    # Use Eq.30's warmup on ALL physics terms, Eq.27 squared cosine, and
    # Algorithm2's mean reductions/clamp10. No labels mask the physics losses.
    terms = {"classification": classification.clamp(0, 10),
             "reconstruction": reconstruction.clamp(0, 10),
             "smoothness": smoothness.clamp(0, 10), "orthogonality": orthogonality.clamp(0, 10)}
    total = terms["classification"] + alpha * (
        cfg.reconstruction_weight * terms["reconstruction"]
        + cfg.smoothness_weight * terms["smoothness"] + cfg.orthogonality_weight * terms["orthogonality"])
    return total, terms
