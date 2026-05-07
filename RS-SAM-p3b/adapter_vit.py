"""
Adapter-ViT with deep DSM fusion (inspired by MFNet MMAdapter).

Route A (basic): adapter prompt_learn before each frozen ViT block
Route A+DSM (deep): adds DSM feature modulation at each block, mimicking
    MFNet's progressive cross-modal fusion inside the encoder.

Two modes:
- AdapterBlock: RGB-only adapter (same as p3)
- AdapterBlockDSM: RGB+DSM adapter with per-block DSM projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterBlock(nn.Module):
    """Wrap a frozen ViT Block with trainable prompt_learn (VPT-style)."""

    def __init__(self, block: nn.Module, dim: int = 1024, bottleneck: int = 32):
        super().__init__()
        self.block = block
        for p in self.block.parameters():
            p.requires_grad = False

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


class AdapterBlockDSM(AdapterBlock):
    """Cross-attention DSM fusion (inspired by MFNet MMAdapter).

    RGB tokens query DSM tokens via cross-attention, producing context-aware
    elevation modulation. DSM tokens are pooled to a fixed grid (dsm_grid²)
    to keep cross-attention memory bounded.

    This is a substantial upgrade over token addition/gating:
    - RGB tokens learn WHICH DSM regions to attend to for each spatial location
    - Multi-head cross-attention with 4 heads, dim head_dim=64
    - DSM context is computed as: attn(Q_rgb, K_dsm) @ V_dsm
    """

    def __init__(self, block: nn.Module, dim: int = 1024, bottleneck: int = 32,
                 dsm_dim: int = 128, dsm_grid: int = 14, num_heads: int = 4):
        super().__init__(block, dim=dim, bottleneck=bottleneck)
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dsm_grid = dsm_grid

        # DSM → QKV projections
        self.dsm_q_proj = nn.Linear(dim, dim)   # RGB tokens as queries
        self.dsm_k_proj = nn.Linear(dsm_dim, dim)  # DSM tokens as keys
        self.dsm_v_proj = nn.Linear(dsm_dim, dim)  # DSM tokens as values
        self.dsm_out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 1024) ViT tokens."""
        prompt_rgb = self.prompt_learn(x)
        prompt = prompt_rgb

        dsm_map = getattr(self, '_dsm_map', None)
        if dsm_map is not None and x.shape[1] >= 64:
            B, N, C = prompt_rgb.shape
            # Pool DSM 2D map to fixed grid for efficient cross-attention
            dsm_pooled = F.adaptive_avg_pool2d(dsm_map, (self.dsm_grid, self.dsm_grid))
            dsm_tokens = dsm_pooled.flatten(2).transpose(1, 2)  # (B, G², dsm_dim)
            M = dsm_tokens.shape[1]

            # Multi-head cross-attention: RGB → DSM
            q = self.dsm_q_proj(prompt_rgb).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, hd)
            k = self.dsm_k_proj(dsm_tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, M, hd)
            v = self.dsm_v_proj(dsm_tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            dsm_context = (attn @ v).transpose(1, 2).reshape(B, N, C)
            prompt = prompt_rgb + self.dsm_out_proj(dsm_context)

        return self.block(x + prompt)


class DSMEncoderDeep(nn.Module):
    """Multi-scale DSM encoder producing features at each ViT stage.

    Unlike the simple DSMEncoder in p3 (which only produces 3 FPN-matched scales),
    this encoder produces per-block DSM tokens for progressive fusion.
    """

    def __init__(self, dsm_dim: int = 128, num_blocks: int = 32,
                 stages: tuple = (8, 16, 24, 32)):
        super().__init__()
        self.dsm_dim = dsm_dim
        self.stages = stages  # block indices where spatial resolution changes

        # Initial DSM encoding: 1ch → dsm_dim channels, stride 16 to match ViT tokens
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, dsm_dim, 3, stride=2, padding=1), nn.BatchNorm2d(dsm_dim), nn.ReLU(True),
        )
        # After 4× stride-2: input / 16 resolution → matches ViT patch embedding

        # Stage transition convolutions (downsample at resolution changes)
        self.transitions = nn.ModuleList()
        prev_s = 0
        for s in stages[:-1]:  # stages where resolution halves
            self.transitions.append(
                nn.Sequential(
                    nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(dsm_dim), nn.ReLU(True)))

    def forward(self, dsm: torch.Tensor) -> dict:
        """dsm: (B, 1, H, W) or (B, H, W)
        Returns: dict with 'features': 2D (B, dsm_dim, H/16, W/16),
                 'scales': [3 FPN-matched scales]
        """
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        x = self.stem(dsm)  # (B, dsm_dim, H/16, W/16)

        # FPN-matched scales
        scales = [F.interpolate(x, (s, s), mode='bilinear', align_corners=False)
                  for s in [288, 144, 72]]

        return {'features': x, 'scales': scales}


import torch.nn.functional as F


def inject_adapters(vision_backbone: nn.Module, bottleneck: int = 32,
                    use_dsm: bool = True) -> nn.Module:
    """Replace all ViT blocks with AdapterBlock or AdapterBlockDSM wrappers.

    Args:
        vision_backbone: SAM3's vision_backbone (Sam3DualViTDetNeck)
        bottleneck: adapter bottleneck dim
        use_dsm: if True, use AdapterBlockDSM for deep DSM fusion

    Returns:
        Modified vision_backbone (in-place)
    """
    blocks = vision_backbone.trunk.blocks
    # Handle already-wrapped blocks (e.g. from parent class init)
    b0 = blocks[0]
    if isinstance(b0, (AdapterBlock, AdapterBlockDSM)):
        dim = b0.block.attn.qkv.in_features
    else:
        dim = b0.attn.qkv.in_features  # 1024
    replaced = 0

    cls = AdapterBlockDSM if use_dsm else AdapterBlock

    for i, blk in enumerate(blocks):
        if not isinstance(blk, cls):
            # Unwrap old adapter if re-wrapping
            inner = blk.block if isinstance(blk, (AdapterBlock, AdapterBlockDSM)) else blk
            blocks[i] = cls(inner, dim=dim, bottleneck=bottleneck)
            replaced += 1

    print(f"  Injected {replaced} {cls.__name__}s (dim={dim}, bottleneck={bottleneck})")

    # Freeze everything except adapter params
    for name, p in vision_backbone.named_parameters():
        if any(k in name for k in ('prompt_learn', 'dsm_proj', 'gate_mlp')):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(f"  Vision backbone: {trainable:,} / {total:,} trainable ({trainable/total*100:.1f}%)")

    return vision_backbone
