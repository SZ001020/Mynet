"""Cross-Attention Adapter for DSM->RGB interaction (Chain 1, Phase A).

Unidirectional cross-attention: DSM tokens query RGB tokens.
Inserted only at global attention blocks [7, 15, 23, 31].

Config (P8-1 recommended):
  - num_heads=8, head_dim=64, proj_dim=512
  - Pre-norm with block.norm1 before Q/K/V projection
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionAdapter(nn.Module):
    """DSM->RGB cross-attention: DSM tokens query RGB key/value representations.

    Q = to_q(norm(dsm))  -- DSM serves as query
    K = to_k(norm(rgb))  -- RGB serves as key
    V = to_v(norm(rgb))  -- RGB serves as value

    Args:
        dim: Token dimension (1024 for SAM3 ViTDet).
        num_heads: Number of attention heads (8, half of SAM3's 16).
        head_dim: Dimension per head (64, standard ViT convention).
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        proj_dim = num_heads * head_dim  # 512
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, proj_dim)
        self.to_k = nn.Linear(dim, proj_dim)
        self.to_v = nn.Linear(dim, proj_dim)
        self.to_out = nn.Linear(proj_dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, y_norm: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """Cross-attn: DSM (Q) queries RGB (K, V).

        Args:
            y_norm: Normalized DSM tokens [B, H, W, dim].
            x_norm: Normalized RGB tokens [B, H, W, dim].

        Returns:
            Cross-attention output [B, H, W, dim] to be added to DSM residual.
        """
        B, H, W, D = y_norm.shape
        N = H * W

        q = self.to_q(y_norm).reshape(B, N, self.num_heads, self.head_dim)
        k = self.to_k(x_norm).reshape(B, N, self.num_heads, self.head_dim)
        v = self.to_v(x_norm).reshape(B, N, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # flash-attn, O(N) memory

        out = out.permute(0, 2, 1, 3).reshape(B, H, W, -1)  # [B, H, W, proj_dim]
        return self.to_out(out)  # [B, H, W, dim]
