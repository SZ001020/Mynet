"""DSM slope/edge/curvature prompt encoder for Plan7-B3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each sample/channel by its p95 and clamp to [0, 1]."""

    b, c = x.shape[:2]
    flat = x.flatten(2)
    scale = torch.quantile(flat.detach().float(), 0.95, dim=-1).view(b, c, 1, 1)
    return (x / (scale.to(x.dtype) + eps)).clamp(0.0, 1.0)


def dsm_prompt_channels(dsm: torch.Tensor) -> torch.Tensor:
    """Create three prompt channels from DSM: slope, edge, and local curvature."""

    if dsm.dim() == 3:
        dsm = dsm.unsqueeze(1)
    dsm = dsm.float()

    sobel_x = dsm.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    sobel_y = dsm.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    lap = dsm.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)

    dx = F.conv2d(dsm, sobel_x, padding=1)
    dy = F.conv2d(dsm, sobel_y, padding=1)
    slope = torch.sqrt(dx.square() + dy.square() + 1e-8)
    edge = F.conv2d(dsm, lap, padding=1).abs()
    curvature = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
    return robust_unit(torch.cat([slope, edge, curvature], dim=1))


class PromptEncoder(nn.Module):
    """Encode DSM slope/edge/curvature prompt to SAM3 ViT patch tokens."""

    def __init__(self, token_dim: int = 1024, prompt_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, prompt_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(prompt_dim, prompt_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(prompt_dim, token_dim, 1)

    def forward(self, dsm: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        prompt = dsm_prompt_channels(dsm)
        feat = self.proj(self.stem(prompt))
        if feat.shape[-2:] != spatial_hw:
            feat = F.interpolate(feat, spatial_hw, mode="bilinear", align_corners=False)
        return feat.permute(0, 2, 3, 1).contiguous()
