"""SimpleDecoder: 4× bilinear upsampling blocks, no GLA attention, no WF fusion.

Used as the shared decoder for all Chain 1 ablation experiments.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__(nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False))


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, decode_channels: int = 64, num_classes: int = 5):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, 1)
        self.up_blocks = nn.ModuleList([
            nn.Sequential(ConvBNReLU(decode_channels, decode_channels, 3),
                          ConvBNReLU(decode_channels, decode_channels, 3))
            for _ in range(4)
        ])
        self.head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels, 3),
            nn.Conv2d(decode_channels, num_classes, 1),
        )

    def forward(self, x):
        x = self.pre_conv(x)
        for up in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up(x)
        return self.head(x)
