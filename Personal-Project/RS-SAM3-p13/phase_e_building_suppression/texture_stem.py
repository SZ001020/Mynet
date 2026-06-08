"""Lightweight CNN for raw RGB texture feature extraction.

Operates at pixel level to capture fine texture patterns (grain, local variance,
edge density) that SAM3 ViT's 14x14 patch embedding abstracts away.
Outputs features at 1/4 input resolution for fusion with decoder features.
"""

import torch.nn as nn


class TextureStem(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_ch, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, rgb):
        return self.stem(rgb)
