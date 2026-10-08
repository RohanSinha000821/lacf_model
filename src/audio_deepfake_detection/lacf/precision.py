"""BF16 matrix/convolution work with explicit FP32 normalization boundaries."""

import torch
from torch import nn


class _FP32Normalization:
    def forward(self, inputs):
        with torch.autocast(inputs.device.type, enabled=False):
            return super().forward(inputs.float())


class FP32LayerNorm(_FP32Normalization, nn.LayerNorm):
    pass


class FP32GroupNorm(_FP32Normalization, nn.GroupNorm):
    pass


class FP32BatchNorm2d(_FP32Normalization, nn.BatchNorm2d):
    pass


def keep_normalization_fp32(module):
    """Preserve normalization state names/values; change only compute precision."""
    for name, child in module.named_children():
        if type(child) is nn.LayerNorm:
            replacement = FP32LayerNorm(child.normalized_shape, eps=child.eps,
                                        elementwise_affine=child.elementwise_affine,
                                        bias=child.bias is not None)
        elif type(child) is nn.GroupNorm:
            replacement = FP32GroupNorm(child.num_groups, child.num_channels,
                                        eps=child.eps, affine=child.affine)
        elif type(child) is nn.BatchNorm2d:
            replacement = FP32BatchNorm2d(child.num_features, eps=child.eps, momentum=child.momentum,
                                          affine=child.affine, track_running_stats=child.track_running_stats)
        else:
            keep_normalization_fp32(child)
            continue
        replacement.load_state_dict(child.state_dict())
        replacement.train(child.training)
        setattr(module, name, replacement)


def fp32_projection_output(module, inputs, output):
    # CLAP get_audio_features normalizes immediately after this projection.
    # Cast before that internal normalization, while retaining BF16 projection work.
    return output.float()
