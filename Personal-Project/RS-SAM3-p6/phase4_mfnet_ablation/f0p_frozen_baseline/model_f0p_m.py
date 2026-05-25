"""F0'+M: SAM3 + LoRA(attn q,v) + MMLoRA dual-branch λ mixing per ViT block.

MMLoRA architecture on SAM3: encodes RGB and DSM in a SINGLE forward pass
with per-block cross-modal λ mixing after MLP.
"""

from __future__ import annotations
import sys, os, math

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
sys.path.extend([SE, SHARED, PHASE1])

import torch, torch.nn as nn, torch.nn.functional as F
from se_fusion import SEFusion
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from lora_layers import inject_lora_into_module


class MMLoRAMixing(nn.Module):
    """MMLoRA per-block mixing: LoRA decomposition + λ-weighted cross-modal fusion."""

    def __init__(self, dim, rank=8):
        super().__init__()
        self.Ax = nn.Linear(dim, rank, bias=False)
        self.Bx = nn.Linear(rank, dim, bias=False)
        self.Ay = nn.Linear(dim, rank, bias=False)
        self.By = nn.Linear(rank, dim, bias=False)
        self.lambda1 = nn.Parameter(torch.tensor(2.0))  # sigmoid(2)≈0.88, start own-modal dominant
        self.lambda2 = nn.Parameter(torch.tensor(2.0))

        nn.init.kaiming_uniform_(self.Ax.weight, a=math.sqrt(5))
        nn.init.zeros_(self.Bx.weight)
        nn.init.kaiming_uniform_(self.Ay.weight, a=math.sqrt(5))
        nn.init.zeros_(self.By.weight)

    def forward(self, x, y):
        x_ada = self.Bx(self.Ax(x))
        y_ada = self.By(self.Ay(y))
        lam1 = torch.sigmoid(self.lambda1)
        lam2 = torch.sigmoid(self.lambda2)
        return x + lam1 * x_ada + (1 - lam1) * y_ada, \
               y + lam2 * y_ada + (1 - lam2) * x_ada


class PassThroughBlock(nn.Module):
    """Thin wrapper: passes DSM tokens through a regular ViT block alongside RGB."""

    def __init__(self, block, state):
        super().__init__()
        self.block = block
        self.state = state

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        y = self.state.get("dsm_tokens")
        if y is not None:
            y_out = checkpoint(self.block, y, use_reentrant=False)
            self.state["dsm_tokens"] = y_out
        return checkpoint(self.block, x, use_reentrant=False)


class MMLoRABlock(nn.Module):
    """ViT block + MMLoRA: shared attention+MLP (with LoRA on attn) + λ mixing after MLP."""

    def __init__(self, block, dim, state, rank=8):
        super().__init__()
        self.block = block
        self.state = state
        self.mixing = MMLoRAMixing(dim, rank)
        self.window_size = getattr(block, "window_size", 0)

    def _match_hw(self, tokens, target_hw):
        if tokens.shape[1:3] == target_hw:
            return tokens
        return F.interpolate(
            tokens.permute(0, 3, 1, 2), target_hw,
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        y = self.state.get("dsm_tokens")
        if y is None:
            return self.block(x)

        hw = x.shape[1:3]
        y = self._match_hw(y, hw)

        # Checkpoint each block forward to avoid holding both activation sets
        from torch.utils.checkpoint import checkpoint
        x_out = checkpoint(self.block, x, use_reentrant=False)
        y_out = checkpoint(self.block, y, use_reentrant=False)

        # MMLoRA mixing (lightweight, activations kept)
        mix_x, mix_y = self.mixing(x, y)
        x_out = x_out + mix_x
        y_out = y_out + mix_y

        self.state["dsm_tokens"] = y_out
        return x_out


class DSMLightEncoder(nn.Module):
    """Lightweight DSM token encoder (matching Plan6/7 DSMTokenEncoder)."""

    def __init__(self, token_dim=1024, dsm_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
            nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(dsm_dim, token_dim, 1)

    def forward(self, dsm, patch_hw):
        feat = self.stem(dsm)
        feat = F.interpolate(feat, patch_hw, mode="bilinear", align_corners=False)
        tokens = self.proj(feat).permute(0, 2, 3, 1).contiguous()
        return tokens


def inject_mmlora(vision_backbone, rank=8):
    """Replace selected ViTDet blocks with MMLoRA-wrapped versions.

    Only wraps global attention blocks [7, 15, 23, 31] to control memory.
    Regular window-attn blocks pass through unchanged.
    """
    trunk = vision_backbone.trunk
    trunk.use_act_checkpoint = False
    blocks = trunk.blocks
    first = blocks[0]
    dim = first.attn.qkv.in_features
    state = {}
    global_blocks = {7, 15, 23, 31}
    replaced = 0
    for idx, block in enumerate(blocks):
        if idx in global_blocks and not isinstance(block, MMLoRABlock):
            blocks[idx] = MMLoRABlock(block, dim, state, rank)
            replaced += 1
        elif idx not in global_blocks and not isinstance(block, PassThroughBlock):
            blocks[idx] = PassThroughBlock(block, state)
    for n, p in vision_backbone.named_parameters():
        p.requires_grad = any(k in n for k in ("mixing", "lora_"))
    print(f"  Injected {replaced} MMLoRA blocks at global attn positions (rank={rank})")
    return state


class FrozenSAM3MMLoRA(nn.Module):
    """F0'+M: SAM3 + MMLoRA (LoRA attn + per-block λ mixing) + DFM.

    Single encoder forward pass processing both RGB and DSM simultaneously.
    """

    def __init__(self, sam3_model, num_classes=5, decode_channels=64,
                 dropout=0.1, resolution=1008, lora_rank=8, mixing_rank=8):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution

        # 1. Inject MMLoRA blocks
        self.mm_state = inject_mmlora(self.backbone.vision_backbone, rank=mixing_rank)

        # 2. LoRA on attn qkv/proj only
        ATTN = [r'attn\.qkv$', r'attn\.proj$']
        n_lora = inject_lora_into_module(self.backbone.vision_backbone.trunk, rank=lora_rank, alpha=16.0, target_patterns=ATTN)
        print(f"  LoRA(attn-only) injected: {n_lora} layers (rank={lora_rank})")

        # 3. Light DSM encoder (feeds tokens into MMLoRA blocks)
        self.dsm_encoder = DSMLightEncoder(token_dim=1024, dsm_dim=128)

        # 4. DFM + decoder
        self.pyramid_x = Pyramid4Scale(256)
        self.pyramid_y = Pyramid4Scale(256)
        self.fusion4 = SEFusion(256); self.fusion3 = SEFusion(256)
        self.fusion2 = SEFusion(256); self.fusion1 = SEFusion(256)
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=decode_channels, dropout=dropout)

        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  F0'+M model: trainable={trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        patch_hw = (self.resolution // 16, self.resolution // 16)

        # RGB forward with DSM tokens injected via MMLoRA blocks
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        try:
            backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)

        deepx = backbone_out["backbone_fpn"][-1]

        # DSM separate pass for DFM's second pyramid input
        dsm_3ch = dsm.repeat(1, 3, 1, 1)
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        fx = self.pyramid_x(deepx); fy = self.pyramid_y(deepy)
        f4 = self.fusion4(fx[0], fy[0]); f3 = self.fusion3(fx[1], fy[1])
        f2 = self.fusion2(fx[2], fy[2]); f1 = self.fusion1(fx[3], fy[3])
        return self.decoder([f4, f3, f2, f1])
