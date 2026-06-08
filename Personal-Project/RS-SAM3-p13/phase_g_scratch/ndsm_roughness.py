"""nDSM roughness features for P13-G.

These features use local structure instead of absolute height:
local variance, max-min roughness, slope magnitude, and morphology residuals.
They are differentiable approximations built from pooling operators.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _avg_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)


def _max_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)


def _min_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)


class NDSMRoughnessStem(nn.Module):
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(10, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def _features(self, ndsm: torch.Tensor) -> torch.Tensor:
        if ndsm.dim() == 3:
            ndsm = ndsm.unsqueeze(1)
        feats = [ndsm]

        for k in (3, 7, 11):
            mean = _avg_pool_same(ndsm, k)
            var = _avg_pool_same(ndsm * ndsm, k) - mean * mean
            rough = _max_pool_same(ndsm, k) - _min_pool_same(ndsm, k)
            feats.extend([var.clamp_min(0.0), rough])

        grad_x = F.conv2d(ndsm, self.sobel_x.to(dtype=ndsm.dtype), padding=1)
        grad_y = F.conv2d(ndsm, self.sobel_y.to(dtype=ndsm.dtype), padding=1)
        slope = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        opened = _max_pool_same(_min_pool_same(ndsm, 7), 7)
        closed = _min_pool_same(_max_pool_same(ndsm, 7), 7)
        feats.extend([slope, (ndsm - opened).abs(), (closed - ndsm).abs()])
        return torch.cat(feats, dim=1)

    def forward(self, ndsm: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        feats = self.proj(self._features(ndsm))
        if feats.shape[-2:] != out_size:
            feats = F.interpolate(feats, out_size, mode="bilinear", align_corners=False)
        return feats
