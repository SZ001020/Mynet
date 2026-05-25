"""Plan6 Phase 1: MFNet-style in-ViT RGB/DSM MMAdapter for SAM3.

This module keeps SAM3's original ViTDet weights frozen and injects trainable
low-rank multimodal adapters around each frozen ViT block. A shared state object
passes the DSM token stream from block to block, so RGB and DSM interact inside
the ViT rather than only in the decoder.
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
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim),
            nn.ReLU(inplace=True),
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
    """Wrap one frozen SAM3 ViT block with MFNet-style RGB/DSM adapters."""

    def __init__(
        self,
        block: nn.Module,
        state: Dict[str, torch.Tensor],
        dim: int = 1024,
        bottleneck: int = 32,
        dsm_attn_mode: str = "adapter",
        checkpoint_attn: bool = False,
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

        self.rgb_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.rgb_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_mlp_adapter = LowRankAdapter(dim, bottleneck)

        self.softmax_logits = nn.Parameter(torch.zeros(2))  # B1: 2-value softmax gate
        self.window_size = getattr(block, "window_size", 0)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen block attention branch on BHWC tokens."""
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
            # Full DSM self-attention is the faithful MFNet-style dual stream.
            # Checkpointing recomputes frozen attention in backward to keep this
            # feasible on 32GB GPUs, especially at SAM3's 1008px resolution.
            attn_y = self._attend_maybe_checkpoint(y)
        else:
            # Lightweight mode avoids a second full SAM3 attention pass. It is
            # faster and safer for batch>1 but has weaker DSM token modeling.
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

        wx = F.softmax(self.softmax_logits, dim=0)[0]
        wy = F.softmax(self.softmax_logits, dim=0)[1]
        x = x + self.block.dropout(self.block.drop_path(mlp_x + wx * adax + (1.0 - wx) * aday))
        y = y + self.block.dropout(self.block.drop_path(mlp_y + wy * aday + (1.0 - wy) * adax))

        self.state["dsm_tokens"] = y
        return x


def inject_mm_adapters(
    vision_backbone: nn.Module,
    bottleneck: int = 32,
    dsm_attn_mode: str = "adapter",
    checkpoint_attn: bool = False,
) -> Dict[str, torch.Tensor]:
    """Replace SAM3 ViT blocks in-place with Plan6 MMAdapter wrappers."""
    trunk = vision_backbone.trunk
    trunk.use_act_checkpoint = False  # shared DSM state is not checkpoint-safe
    blocks = trunk.blocks
    first = blocks[0].block if isinstance(blocks[0], MMAdapterBlock) else blocks[0]
    dim = first.attn.qkv.in_features
    state: Dict[str, torch.Tensor] = {}

    replaced = 0
    for idx, block in enumerate(blocks):
        inner = block.block if isinstance(block, MMAdapterBlock) else block
        if not isinstance(block, MMAdapterBlock):
            blocks[idx] = MMAdapterBlock(
                inner,
                state=state,
                dim=dim,
                bottleneck=bottleneck,
                dsm_attn_mode=dsm_attn_mode,
                checkpoint_attn=checkpoint_attn,
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
                "softmax_logits",
            )
        )

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(
        f"  Injected {replaced} Plan6 MMAdapter blocks "
        f"(dim={dim}, bottleneck={bottleneck}, dsm_attn={dsm_attn_mode}, "
        f"checkpoint_attn={checkpoint_attn})"
    )
    print(f"  Vision backbone trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
    return state
