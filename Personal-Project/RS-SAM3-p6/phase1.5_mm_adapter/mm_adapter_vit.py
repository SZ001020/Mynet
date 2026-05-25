"""Plan6 Phase 1: MFNet-style in-ViT RGB/DSM MMAdapter for SAM3.

This module keeps SAM3's original ViTDet weights frozen and injects trainable
low-rank multimodal adapters around each frozen ViT block. A shared state object
passes the DSM token stream from block to block, so RGB and DSM interact inside
the ViT rather than only in the decoder.
"""

from __future__ import annotations

from typing import Dict
from contextlib import contextmanager

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


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen Linear layer.

    The original projection stays frozen. LoRA starts as a no-op because B is
    zero-initialized, so Phase 1 checkpoints can be loaded safely.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True

        for param in self.original.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, original.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        if self.enabled:
            out = out + (x @ self.lora_A @ self.lora_B) * self.scaling
        return out


@contextmanager
def lora_enabled(module: nn.Module, enabled: bool):
    """Temporarily enable/disable LoRA wrappers under a module."""

    wrappers = [m for m in module.modules() if isinstance(m, LoRALinear)]
    old = [m.enabled for m in wrappers]
    for wrapper in wrappers:
        wrapper.enabled = enabled
    try:
        yield
    finally:
        for wrapper, value in zip(wrappers, old):
            wrapper.enabled = value


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
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        if dsm_attn_mode not in {"adapter", "full"}:
            raise ValueError(f"Unsupported dsm_attn_mode: {dsm_attn_mode}")
        self.block = block
        self.state = state
        self.dsm_attn_mode = dsm_attn_mode
        self.checkpoint_attn = checkpoint_attn
        self.lora_rank = lora_rank
        for param in self.block.parameters():
            param.requires_grad = False

        if lora_rank > 0:
            inject_lora_into_block(self.block, rank=lora_rank, alpha=lora_alpha)

        self.rgb_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.rgb_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_mlp_adapter = LowRankAdapter(dim, bottleneck)

        self.wx_logit = nn.Parameter(torch.zeros(1))
        self.wy_logit = nn.Parameter(torch.zeros(1))
        self.window_size = getattr(block, "window_size", 0)

    def _attend(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        """Run the frozen block attention branch on BHWC tokens."""
        with lora_enabled(self.block.attn, use_lora):
            x = self.block.norm1(x)
            if self.window_size > 0:
                h, w = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)
            x = self.block.ls1(self.block.attn(x))
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (h, w))
            return x

    def _attend_maybe_checkpoint(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        if self.checkpoint_attn and self.training and x.requires_grad:
            return checkpoint(lambda t: self._attend(t, use_lora=use_lora), x, use_reentrant=False)
        return self._attend(x, use_lora=use_lora)

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

        attn_x = self._attend_maybe_checkpoint(x, use_lora=True)
        if self.dsm_attn_mode == "full":
            # Full DSM self-attention is the faithful MFNet-style dual stream.
            # Checkpointing recomputes frozen attention in backward to keep this
            # feasible on 32GB GPUs, especially at SAM3's 1008px resolution.
            # The LoRA correction is reserved for the RGB backbone path. DSM
            # still uses frozen SAM3 attention plus MMAdapter fusion, avoiding
            # the old failure mode where LoRA becomes a single-modal shortcut.
            attn_y = self._attend_maybe_checkpoint(y, use_lora=False)
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
        with lora_enabled(self.block.mlp, True):
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
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
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
                "lora_A",
                "lora_B",
            )
        )

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(
        f"  Injected {replaced} Plan6 MMAdapter blocks "
        f"(dim={dim}, bottleneck={bottleneck}, dsm_attn={dsm_attn_mode}, "
        f"checkpoint_attn={checkpoint_attn}, lora_rank={lora_rank})"
    )
    print(f"  Vision backbone trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
    return state


def inject_lora_into_block(block: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Inject LoRA into SAM3 ViT attention and MLP projections."""

    injected = 0
    targets = [
        (getattr(block, "attn", None), "qkv"),
        (getattr(block, "attn", None), "proj"),
        (getattr(block, "mlp", None), "fc1"),
        (getattr(block, "mlp", None), "fc2"),
    ]
    for parent, attr in targets:
        if parent is None or not hasattr(parent, attr):
            continue
        layer = getattr(parent, attr)
        if isinstance(layer, LoRALinear):
            continue
        if isinstance(layer, nn.Linear):
            setattr(parent, attr, LoRALinear(layer, rank=rank, alpha=alpha))
            injected += 1
    return injected
