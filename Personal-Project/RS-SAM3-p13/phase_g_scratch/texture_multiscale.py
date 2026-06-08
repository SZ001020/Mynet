"""Multi-scale RGB texture stem for P13-G.

The branches mirror the literature-driven texture windows:
3x3 for fine grass texture, 7x7 for mixed vegetation, and 11x11 for
coarser tree crown texture. The stem stays lightweight and feeds only the
vegetation binary head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TextureBranch(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                stride=2,
                padding=pad,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleTextureStem(nn.Module):
    def __init__(self, in_ch: int = 3, branch_ch: int = 48, out_ch: int = 128):
        super().__init__()
        self.branch3 = _TextureBranch(in_ch, branch_ch, 3)
        self.branch7 = _TextureBranch(in_ch, branch_ch, 7)
        self.branch11 = _TextureBranch(in_ch, branch_ch, 11)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, rgb: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        feats = torch.cat([self.branch3(rgb), self.branch7(rgb), self.branch11(rgb)], dim=1)
        feats = self.fuse(feats)
        if feats.shape[-2:] != out_size:
            feats = F.interpolate(feats, out_size, mode="bilinear", align_corners=False)
        return feats
