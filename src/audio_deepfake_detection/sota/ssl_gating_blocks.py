"""Selected released MultiConv/pooling operations, without ESPnet dependencies.

Adapted from hoanmyTran/dissimilarity_deepfake_detection at 4de821c.
MultiConv derives from ESPnet; attentive pooling derives from asv-subtools
(Copyright xmuspeech, Author: Snowdar 2019-05-29 2020-06-10).
See SSL_GATING_LICENSE and SSL_GATING_APACHE_LICENSE for retained notices.
Only the authors' configured concat_fusion and four-head pooling paths remain.
"""

import torch
from torch import nn


class MultiConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_proj1 = nn.Sequential(nn.Linear(128, 1024), nn.GELU())
        # ESPnet LayerNorm uses eps=1e-12, unlike torch's default 1e-5.
        self.norm = nn.LayerNorm(512, eps=1e-12)
        self.convs = nn.ModuleList([
            nn.Conv1d(512, 128, k, padding=(k - 1) // 2, groups=128)
            for k in (3, 7, 11, 15)
        ])
        self.fusion = nn.Conv1d(512, 512, 15, padding=7, groups=512)
        self.linear = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.1)
        self.channel_proj2 = nn.Linear(512, 128)

    def forward(self, x):
        residual, gate = self.channel_proj1(x).chunk(2, dim=-1)
        gate = self.norm(gate).transpose(1, 2)
        gate = torch.cat([conv(gate).transpose(1, 2) for conv in self.convs], dim=-1)
        gate = gate + self.fusion(gate.transpose(1, 2)).transpose(1, 2)
        gate = torch.nn.functional.silu(self.linear(gate))
        return self.channel_proj2(self.dropout(residual * gate))


class MultiHeadPooling(nn.Module):
    def __init__(self):
        super().__init__()
        # Released TdnnAffine: one independent 128-channel head per block,
        # no bias, Gaussian(0, .01) weights, not Conv1d's default initialization.
        self.attention = nn.Conv1d(512, 4, 1, groups=4, bias=False)
        nn.init.normal_(self.attention.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """[B,T,512] -> pooled [B,1024], means first then standard deviations."""
        x = x.transpose(1, 2)
        batch, _, frames = x.shape
        alpha = self.attention(x).softmax(dim=-1).reshape(batch, 4, 1, frames)
        split = x.reshape(batch, 4, 128, frames)
        mean = (alpha * split).sum(dim=-1).reshape(batch, 512)
        variance = (alpha * split.square()).sum(dim=-1).reshape(batch, 512) - mean.square()
        return torch.cat((mean, variance.clamp_min(1e-10).sqrt()), dim=-1)
