"""Plan8 Chain 2 Phase A: in-ViT MMAdapter with DSM elevation-difference attention bias.

Extends Plan7-A's MMAdapterPromptBlock by injecting a factorized DSM elevation
bias into RGB self-attention scores at global blocks [7, 15, 23, 31].

The bias is added to Q@K^T/sqrt(d) before softmax, modulating how RGB tokens
attend to each other based on per-token DSM elevation information.

Key changes from Plan7-A:
  - _attend_with_bias(x, y): re-implements attention manually for global blocks
    to inject the bias between Q@K^T and softmax (fused SDPA can't be intercepted).
  - DSM self-attention (attn_y) is unchanged (no bias applied).
  - Window attention blocks use the original _attend (unchanged).
"""

from __future__ import annotations

from typing import Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sam3.model.vitdet import window_partition, window_unpartition

from attn_bias import DSMAttentionBias

GLOBAL_BLOCK_IDS: Set[int] = {7, 15, 23, 31}


class LowRankAdapter(nn.Module):
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
    """DSM raw height map -> ViT token space.

    Identical to Plan7-A DSMTokenEncoder. Defined here so chain2 is self-contained.
    """

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


class MMAdapterPromptBlockAB(nn.Module):
    """Frozen SAM3 block with RGB/DSM/prompt adapter + DSM attention bias.

    The bias only applies to RGB self-attention at global blocks.
    Gate fusion mechanism is identical to Plan7-A.
    """

    def __init__(
        self,
        block: nn.Module,
        state: Dict[str, torch.Tensor],
        dim: int = 1024,
        bottleneck: int = 32,
        dsm_attn_mode: str = "full",
        checkpoint_attn: bool = False,
        block_idx: int = 0,
    ):
        super().__init__()
        if dsm_attn_mode not in {"adapter", "full"}:
            raise ValueError(f"Unsupported dsm_attn_mode: {dsm_attn_mode}")
        self.block = block
        self.state = state
        self.dsm_attn_mode = dsm_attn_mode
        self.checkpoint_attn = checkpoint_attn
        self.window_size = getattr(block, "window_size", 0)
        self.block_idx = block_idx
        self.is_global = self.window_size <= 0

        for param in self.block.parameters():
            param.requires_grad = False

        self.rgb_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_attn_adapter = LowRankAdapter(dim, bottleneck)
        self.rgb_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.dsm_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.prompt_mlp_adapter = LowRankAdapter(dim, bottleneck)
        self.rgb_gate_logits = nn.Parameter(torch.zeros(3))
        self.dsm_gate_logits = nn.Parameter(torch.zeros(3))

        # DSM attention bias: only at global blocks
        if block_idx in GLOBAL_BLOCK_IDS:
            self.dsm_attn_bias = DSMAttentionBias(dim=dim, num_heads=block.attn.num_heads)
        else:
            self.dsm_attn_bias = None

    def _attend_original(self, x: torch.Tensor) -> torch.Tensor:
        """Original attention path (for DSM tokens and window blocks)."""
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
            return checkpoint(self._attend_original, x, use_reentrant=False)
        return self._attend_original(x)

    def _attend_with_bias(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Global self-attention on RGB tokens with DSM elevation bias injected.

        Re-implements the attention computation manually (instead of using
        block.attn which calls fused SDPA) to insert the bias between
        Q@K^T and softmax.

        Args:
            x: RGB tokens [B, H, W, D].
            y: DSM tokens [B, H, W, D] (for bias computation).

        Returns:
            Attention output [B, H, W, D].
        """
        B, H, W, D = x.shape
        N = H * W

        x_norm = self.block.norm1(x)  # [B, H, W, D]
        attn_module = self.block.attn

        # Q, K, V from frozen projection
        qkv = attn_module.qkv(x_norm).reshape(B, N, 3, attn_module.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, N, head_dim]

        # Apply rope (SAM3 ViTDet uses RoPE)
        if attn_module.use_rope:
            q, k = attn_module._apply_rope(q, k)

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) * attn_module.scale  # [B, heads, N, N]

        # Inject factorized DSM elevation bias
        bias_h, bias_w, scale = self.dsm_attn_bias(y)
        attn_scores = attn_scores + scale * (bias_h + bias_w)  # broadcast to [B,heads,N,N]

        # Softmax + apply to V
        attn_weights = attn_scores.softmax(dim=-1)
        out = attn_weights @ v  # [B, heads, N, head_dim]

        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, D)  # [B, H, W, D]
        out = attn_module.proj(out)
        out = self.block.ls1(out)

        return out

    def _attend_maybe_checkpoint_biased(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_attn and self.training and x.requires_grad:
            return checkpoint(self._attend_with_bias, x, y, use_reentrant=False)
        return self._attend_with_bias(x, y)

    @staticmethod
    def _match_hw(tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if tokens.shape[1:3] == hw:
            return tokens
        return F.interpolate(
            tokens.permute(0, 3, 1, 2),
            hw,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"AB MMAdapter expects BHWC tokens, got {tuple(x.shape)}")
        hw = x.shape[1:3]
        y = self.state.get("dsm_tokens")
        p = self.state.get("prompt_tokens")
        if y is None or p is None:
            raise RuntimeError("DSM/prompt token state was not set before ViT forward")
        y = self._match_hw(y, hw)
        p = self._match_hw(p, hw)

        # Phase 1a: self-attention
        #   RGB: with DSM elevation bias (global) or regular (window)
        #   DSM: always regular self-attention
        if self.dsm_attn_bias is not None and self.is_global:
            attn_x = self._attend_maybe_checkpoint_biased(x, y)
        else:
            attn_x = self._attend_maybe_checkpoint(x)

        if self.dsm_attn_mode == "full":
            attn_y = self._attend_maybe_checkpoint(y)
        else:
            attn_y = self.dsm_attn_adapter(self.block.norm1(y))

        # Phase 1b: adapter residual (unchanged)
        x = x + self.block.dropout(self.block.drop_path(attn_x + self.rgb_attn_adapter(attn_x)))
        y = y + self.block.dropout(self.block.drop_path(attn_y + self.dsm_attn_adapter(attn_y)))

        # Phase 2: MLP + gate fusion (completely unchanged)
        xn = self.block.norm2(x)
        yn = self.block.norm2(y)
        pn = self.block.norm2(p)
        mlp_x = self.block.ls2(self.block.mlp(xn))
        mlp_y = self.dsm_mlp_adapter(yn)
        rgb_ada = self.rgb_mlp_adapter(xn)
        dsm_ada = self.dsm_mlp_adapter(yn)
        prompt_ada = self.prompt_mlp_adapter(pn)

        wx = torch.softmax(self.rgb_gate_logits, dim=0)
        wy = torch.softmax(self.dsm_gate_logits, dim=0)
        x = x + self.block.dropout(
            self.block.drop_path(mlp_x + wx[0] * rgb_ada + wx[1] * dsm_ada + wx[2] * prompt_ada)
        )
        y = y + self.block.dropout(
            self.block.drop_path(mlp_y + wy[0] * rgb_ada + wy[1] * dsm_ada + wy[2] * prompt_ada)
        )

        self.state["dsm_tokens"] = y
        self.state["prompt_tokens"] = p
        return x


def inject_mm_adapters(
    vision_backbone: nn.Module,
    bottleneck: int = 32,
    dsm_attn_mode: str = "full",
    checkpoint_attn: bool = False,
) -> Dict[str, torch.Tensor]:
    """Replace ViT blocks with MMAdapterPromptBlockAB wrappers.

    Only blocks [7, 15, 23, 31] get DSMAttentionBias.
    """
    trunk = vision_backbone.trunk
    trunk.use_act_checkpoint = False
    blocks = trunk.blocks
    first = blocks[0].block if isinstance(blocks[0], MMAdapterPromptBlockAB) else blocks[0]
    dim = first.attn.qkv.in_features
    state: Dict[str, torch.Tensor] = {}

    bias_blocks = 0
    replaced = 0
    for idx, block in enumerate(blocks):
        inner = block.block if isinstance(block, MMAdapterPromptBlockAB) else block
        if not isinstance(block, MMAdapterPromptBlockAB):
            blocks[idx] = MMAdapterPromptBlockAB(
                inner,
                state=state,
                dim=dim,
                bottleneck=bottleneck,
                dsm_attn_mode=dsm_attn_mode,
                checkpoint_attn=checkpoint_attn,
                block_idx=idx,
            )
            replaced += 1
            if idx in GLOBAL_BLOCK_IDS:
                bias_blocks += 1

    trainable_keys = (
        "rgb_attn_adapter",
        "dsm_attn_adapter",
        "rgb_mlp_adapter",
        "dsm_mlp_adapter",
        "prompt_mlp_adapter",
        "rgb_gate_logits",
        "dsm_gate_logits",
        "dsm_attn_bias",
    )
    for name, param in vision_backbone.named_parameters():
        param.requires_grad = any(key in name for key in trainable_keys)

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(
        f"  Injected {replaced} Plan8-AB prompt MMAdapter blocks "
        f"({bias_blocks} with DSM attn bias) "
        f"(dim={dim}, bottleneck={bottleneck}, dsm_attn={dsm_attn_mode}, "
        f"checkpoint_attn={checkpoint_attn})"
    )
    print(f"  Vision backbone trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
    return state
