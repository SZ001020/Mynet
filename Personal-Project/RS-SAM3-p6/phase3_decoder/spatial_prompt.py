"""Plan6 Phase 3-D2: Learnable Spatial Tokens from DSM injected into MFNetDecoder.

原理来自 TASAM: 从 DSM 特征生成 k 个可学习 spatial prompt tokens，注入 decoder 输入端，
提供位置级辅助信号——"在图像的这些位置注意高度突变"。
与 in-ViT MMAdapter 正交：MMAdapter 在 ViT 内部做语义级 DSM 融合，
spatial tokens 在 decoder 输入端做位置级辅助。
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class SpatialPromptDecoder(nn.Module):
    """从 DSM 生成 learnable spatial tokens，注入 MFNetDecoder。

    DSM feat (B, C, H, W) → AdaptiveAvgPool2d → MLP → (B, num_tokens, C)
    这些 token 被 concat 到 decoder 的 b4 bottleneck 输入中。
    """

    def __init__(self, dsm_dim: int = 128, decode_channels: int = 64,
                 num_tokens: int = 4, token_dim: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dsm_dim, token_dim), nn.GELU(),
            nn.Linear(token_dim, token_dim), nn.GELU(),
        )
        self.token = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)

    def forward(self, dsm_feat: torch.Tensor):
        """dsm_feat: (B, C, H, W) → returns (B, num_tokens, token_dim)"""
        g = self.pool(dsm_feat).flatten(1)  # (B, C)
        g = self.mlp(g).unsqueeze(1)         # (B, 1, token_dim)
        t = self.token.unsqueeze(0) + g      # (B, num_tokens, token_dim)
        return t
