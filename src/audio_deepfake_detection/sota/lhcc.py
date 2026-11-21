"""Paper-informed frozen LHCC adaptation; see scripts/lhcc/README.md for assumptions.

Retains all 24 frame-level XLS-R outputs, asymmetric 3:2 Res2Net streams,
gated fusion, three SE/Transformer blocks and shared first/third attention.
Geometry not established by author code is explicitly a study implementation.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LHCCConfig:
    width: int = 1024
    frames: int = 201  # 64600 samples through the official XLS-R convolutional frontend
    channels: int = 32
    res2_scale: int = 4
    attention_heads: int = 8
    ffn_hidden: int = 1024
    se_reduction: int = 16
    classifier_hidden: int = 1024


class Bottle2neck(nn.Module):
    """Hierarchical Res2Net split convolutions, preserving time/feature coordinates."""
    def __init__(self, channels, scale):
        super().__init__()
        if scale < 2 or channels % scale:
            raise ValueError("Res2Net scale must divide channels")
        self.scale = scale
        self.input = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU())
        self.parts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels // scale, channels // scale, 3, padding=1, bias=False),
                          nn.BatchNorm2d(channels // scale), nn.ReLU()) for _ in range(scale - 1)])
        self.output = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        parts = self.input(x).chunk(self.scale, dim=1)
        outputs = [parts[0]]
        for index, conv in enumerate(self.parts, 1):
            value = parts[index] if index == 1 else parts[index] + outputs[-1]
            outputs.append(conv(value))
        return F.relu(x + self.output(torch.cat(outputs, dim=1)))


class AsymmetricFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        def stream(layers, depth):
            return nn.Sequential(nn.Conv2d(layers, cfg.channels, 1),
                                 nn.MaxPool2d(3, stride=1, padding=1),
                                 *[Bottle2neck(cfg.channels, cfg.res2_scale) for _ in range(depth)])
        self.low, self.high = stream(21, 3), stream(3, 2)
        channels = 2 * cfg.channels
        self.gates = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, cfg.channels, 1),
                                   nn.ReLU(), nn.Conv2d(cfg.channels, channels, 1), nn.Sigmoid())
        self.residual = nn.Conv2d(channels, channels, 1)
        # Explicit study bridge from channel maps to [time,1024], not an author-code claim.
        self.channel_projection = nn.Conv2d(channels, 1, 1)
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, states):
        low, high = self.low(states[:, :21]), self.high(states[:, 21:])
        joined = torch.cat((low, high), dim=1)
        fused = joined * self.gates(joined) + self.residual(joined)
        return self.norm(self.channel_projection(fused).squeeze(1))


class DependencyBlock(nn.Module):
    def __init__(self, cfg, attention):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(cfg.width, max(1, cfg.width // cfg.se_reduction)),
                                 nn.ReLU(), nn.Linear(max(1, cfg.width // cfg.se_reduction), cfg.width),
                                 nn.Sigmoid())
        self.attention = attention
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, cfg.ffn_hidden), nn.ReLU(),
                                  nn.Linear(cfg.ffn_hidden, cfg.width))

    def forward(self, x):
        calibrated = x * self.se(x.mean(dim=1)).unsqueeze(1)
        x = self.norm1(x + self.attention(calibrated, calibrated, calibrated, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class LHCCDetector(nn.Module):
    def __init__(self, cfg=LHCCConfig()):
        super().__init__()
        if min(asdict(cfg).values()) < 1 or cfg.width % cfg.attention_heads or min(cfg.frames, cfg.width) < 3:
            raise ValueError("Invalid LHCC detector geometry")
        self.cfg = cfg
        self.fusion = AsymmetricFusion(cfg)
        shared = nn.MultiheadAttention(cfg.width, cfg.attention_heads, batch_first=True)
        independent = nn.MultiheadAttention(cfg.width, cfg.attention_heads, batch_first=True)
        self.blocks = nn.ModuleList([DependencyBlock(cfg, shared), DependencyBlock(cfg, independent),
                                     DependencyBlock(cfg, shared)])
        self.final_ffn = nn.Sequential(nn.Linear(cfg.width, cfg.ffn_hidden), nn.ReLU(),
                                        nn.Linear(cfg.ffn_hidden, cfg.width))
        # Follow Fig.1's fixed pool/flatten/1024/2 classifier, rather than prose GAP/sigmoid.
        self.pool = nn.AvgPool2d(3, stride=3)
        self.classifier = nn.Sequential(nn.Flatten(),
            nn.Linear((cfg.frames // 3) * (cfg.width // 3), cfg.classifier_hidden), nn.ReLU(),
            nn.Linear(cfg.classifier_hidden, 2))

    def forward(self, states):
        if states.ndim != 4 or tuple(states.shape[1:]) != (24, self.cfg.frames, self.cfg.width):
            raise ValueError("Expected [batch,24,time,width] Transformer outputs in layer order")
        x = self.fusion(states)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.pool(self.final_ffn(x).unsqueeze(1)))

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]


def classification_loss(logits, labels):
    # Two logits implement the paper's weighted binary BCE via their difference.
    # Numeric weights are a declared study choice, not reported LHCC settings.
    losses = F.binary_cross_entropy_with_logits(LHCCDetector.spoof_score(logits).float(),
                                               labels.float(), reduction="none")
    weights = torch.where(labels == 0, 0.9, 0.1)
    return (losses * weights).mean()
