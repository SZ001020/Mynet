"""Plan6 Phase 1.6: MMAdapter + selective attention unfreeze for SAM3 ViTDet.

Extends Phase 1's in-ViT RGB/DSM MMAdapter with the ability to unfreeze
attention weights (qkv, proj) in the deepest ViT blocks, directly attacking
the frozen attention bottleneck without touching shallow generic features.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sam3.model.vitdet import window_partition, window_unpartition


class LowRankAdapter(nn.Module):
    """Small MLP adapter that returns a residual perturbation."""

    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSMTokenEncoder(nn.Module):
    """Encode DSM to the SAM3 ViT patch grid and token dimension."""

    def __init__(self, token_dim: int = 1024, dsm_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim),
            nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(dsm_dim, token_dim, 1)

    def forward(self, dsm: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        feat = self.proj(self.stem(dsm))
        if feat.shape[-2:] != spatial_hw:
            feat = F.interpolate(feat, spatial_hw, mode="bilinear", align_corners=False)
        return feat.permute(0, 2, 3, 1).contiguous()


class MMAdapterBlock(nn.Module):
    """Wrap one frozen SAM3 ViT block with MFNet-style RGB/DSM adapters.

    When unfreeze_attn=True, the block's attention qkv and proj weights are
    trainable with a conservative per-parameter learning rate multiplier.
    MLP weights stay frozen even when unfreeze_attn=True.
    """

    def __init__(
        self,
        block: nn.Module,
        state: Dict[str, torch.Tensor],
        dim: int = 1024,
        bottleneck: int = 32,
        dsm_attn_mode: str = "adapter",
        checkpoint_attn: bool = False,
        unfreeze_attn: bool = False,
    ):
        super().__init__()
        if dsm_attn_mode not in {"adapter", "full"}:
            raise ValueError(f"Unsupported dsm_attn_mode: {dsm_attn_mode}")
        self.block = block
        self.state = state
        self.dsm_attn_mode = dsm_attn_mode
        self.checkpoint_attn = checkpoint_attn

        for param in self.block.parameters():
            param.requires_grad = False

        if unfreeze_attn:
            targets = []
            if hasattr(block.attn, "qkv"):
                targets.append(block.attn.qkv)
            if hasattr(block.attn, "proj"):
                targets.append(block.attn.proj)
            for target in targets:
                for p in target.parameters():
                    p.requires_grad = True

        self.rgb_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.rgb_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_mlp_adapter = LowRankAdapter(dim, bottleneck)

        self.wx_logit = nn.Parameter(torch.zeros(1))
        self.wy_logit = nn.Parameter(torch.zeros(1))
        self.window_size = getattr(block, "window_size", 0)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block.norm1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.block.ls1(self.block.attn(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))
        return x

    def _attend_maybe_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_attn and self.training and x.requires_grad:
            return checkpoint(self._attend, x, use_reentrant=False)
        return self._attend(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Plan6 MMAdapter expects BHWC tokens, got shape {tuple(x.shape)}")

        y = self.state.get("dsm_tokens")
        if y is None:
            raise RuntimeError("DSM token state was not set before SAM3 ViT forward")
        if y.shape[1:3] != x.shape[1:3]:
            y = F.interpolate(
                y.permute(0, 3, 1, 2),
                x.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()

        attn_x = self._attend_maybe_checkpoint(x)
        if self.dsm_attn_mode == "full":
            attn_y = self._attend_maybe_checkpoint(y)
        else:
            attn_y = self.dsm_attn_adapter(self.block.norm1(y))
        x = x + self.block.dropout(
            self.block.drop_path(attn_x + self.rgb_attn_adapter(attn_x))
        )
        y = y + self.block.dropout(
            self.block.drop_path(attn_y + self.dsm_attn_adapter(attn_y))
        )

        xn = self.block.norm2(x)
        yn = self.block.norm2(y)
        mlp_x = self.block.ls2(self.block.mlp(xn))
        mlp_y = self.dsm_mlp_adapter(yn)
        adax = self.rgb_mlp_adapter(xn)
        aday = self.dsm_mlp_adapter(yn)

        wx = torch.sigmoid(self.wx_logit)
        wy = torch.sigmoid(self.wy_logit)
        x = x + self.block.dropout(self.block.drop_path(mlp_x + wx * adax + (1.0 - wx) * aday))
        y = y + self.block.dropout(self.block.drop_path(mlp_y + wy * aday + (1.0 - wy) * adax))

        self.state["dsm_tokens"] = y
        return x


def inject_mm_adapters(
    vision_backbone: nn.Module,
    bottleneck: int = 32,
    dsm_attn_mode: str = "adapter",
    checkpoint_attn: bool = False,
    unfreeze_layers: int = 0,
) -> Dict[str, torch.Tensor]:
    """Replace SAM3 ViT blocks in-place with Plan6 MMAdapter wrappers.

    Args:
        vision_backbone: SAM3 vision_backbone module.
        bottleneck: adapter bottleneck dimension.
        dsm_attn_mode: "adapter" or "full".
        checkpoint_attn: whether to use activation checkpointing.
        unfreeze_layers: number of deepest blocks whose attention weights
            will be trainable. 0 = all frozen (Phase 1 behavior).

    Returns:
        Shared DSM token state dict.
    """
    trunk = vision_backbone.trunk
    trunk.use_act_checkpoint = False
    blocks = trunk.blocks
    first = blocks[0].block if isinstance(blocks[0], MMAdapterBlock) else blocks[0]
    dim = first.attn.qkv.in_features
    state: Dict[str, torch.Tensor] = {}
    num_blocks = len(blocks)

    replaced = 0
    for idx, block in enumerate(blocks):
        inner = block.block if isinstance(block, MMAdapterBlock) else block
        if not isinstance(block, MMAdapterBlock):
            is_deep = (num_blocks - idx) <= unfreeze_layers
            blocks[idx] = MMAdapterBlock(
                inner,
                state=state,
                dim=dim,
                bottleneck=bottleneck,
                dsm_attn_mode=dsm_attn_mode,
                checkpoint_attn=checkpoint_attn,
                unfreeze_attn=is_deep,
            )
            replaced += 1

    for name, param in vision_backbone.named_parameters():
        param.requires_grad = any(
            key in name
            for key in (
                "rgb_attn_adapter",
                "dsm_attn_adapter",
                "rgb_mlp_adapter",
                "dsm_mlp_adapter",
                "wx_logit",
                "wy_logit",
            )
        ) or (
            unfreeze_layers > 0
            and any(
                f"blocks.{num_blocks - 1 - i}" in name
                for i in range(unfreeze_layers)
            )
            and any(k in name for k in ("qkv", "proj"))
        )

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(
        f"  Injected {replaced} Plan6 MMAdapter blocks "
        f"(dim={dim}, bottleneck={bottleneck}, dsm_attn={dsm_attn_mode}, "
        f"checkpoint_attn={checkpoint_attn}, unfreeze_layers={unfreeze_layers})"
    )
    print(f"  Vision backbone trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
    return state
