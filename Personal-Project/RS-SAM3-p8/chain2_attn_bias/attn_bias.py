"""DSM Attention Bias module for Chain 2 Phase A.

Produces a factorized elevation-difference bias that modulates RGB self-attention
scores at global attention blocks. The bias is decomposable into row + column
components to avoid O(N^2) explicit materialization.

Architecture:
  1. elev_proj: Linear(1024->1) extracts a scalar "elevation" from each DSM token
  2. bias_mlp_h / bias_mlp_w: Two MLPs that produce per-head biases from elevation
  3. Factorized: bias[i,j] = bias_h[i] + bias_w[j] (broadcast to [B,heads,N,N])
  4. scale: Learnable scalar controlling bias magnitude
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DSMAttentionBias(nn.Module):
    """Factorized DSM elevation-difference attention bias.

    Physical intuition: tokens with similar elevation (height) should attend more
    to each other. The bias is factorized as row + column for memory efficiency.

    bias[i,j] = scale * (mlp_h(elev_i) + mlp_w(elev_j))
    """

    def __init__(self, dim: int = 1024, num_heads: int = 16):
        """
        Args:
            dim: Token dimension (1024 for SAM3 ViTDet).
            num_heads: Number of attention heads in SAM3 (16, not cross-attn heads).
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.elev_proj = nn.Linear(dim, 1)

        self.bias_mlp_h = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, num_heads),
        )
        self.bias_mlp_w = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, num_heads),
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.elev_proj.weight)
        nn.init.zeros_(self.elev_proj.bias)
        for m in [self.bias_mlp_h, self.bias_mlp_w]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, dsm_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute factorized attention bias from DSM tokens.

        Args:
            dsm_tokens: [B, H, W, dim] DSM token features.

        Returns:
            bias_h: [B, heads, N, 1] row bias component.
            bias_w: [B, heads, 1, N] column bias component.
            scale: Scalar scale factor for the bias.
        """
        B, H, W, D = dsm_tokens.shape
        N = H * W
        x = dsm_tokens.reshape(B, N, D)

        elev = self.elev_proj(x)  # [B, N, 1]

        bias_h = self.bias_mlp_h(elev)  # [B, N, heads]
        bias_w = self.bias_mlp_w(elev)  # [B, N, heads]

        bias_h = bias_h.permute(0, 2, 1).unsqueeze(-1)  # [B, heads, N, 1]
        bias_w = bias_w.permute(0, 2, 1).unsqueeze(-2)  # [B, heads, 1, N]

        return bias_h, bias_w, self.scale
