"""F0'+L: FrozenSAM3DFM + LoRA (rank=8).

Same as F0' but with LoRA injected on SAM3 ViTDet attention layers.
Measures LoRA's net contribution on a clean frozen SAM3 baseline.
"""

from __future__ import annotations
import sys, os

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
sys.path.extend([SE, SHARED, PHASE1])

import torch, torch.nn as nn, torch.nn.functional as F
from se_fusion import SEFusion
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from lora_layers import inject_lora_into_module


class FrozenSAM3DFMLoRA(nn.Module):
    """F0' + LoRA: frozen SAM3 + 4×SEFusion (DFM) + MFNetDecoder + LoRA (rank=8)."""

    def __init__(self, sam3_model, num_classes=5, decode_channels=64,
                 dropout=0.1, resolution=1008, lora_rank=8, lora_alpha=16.0):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution

        # LoRA injection on SAM3 ViTDet
        VISION_PATTERNS = [r'attn\.qkv$', r'attn\.proj$', r'q_proj$', r'k_proj$', r'v_proj$', r'out_proj$',
                           r'mlp\.fc1$', r'mlp\.fc2$']
        trunk = self.backbone.vision_backbone.trunk
        n = inject_lora_into_module(trunk, rank=lora_rank, alpha=lora_alpha, target_patterns=VISION_PATTERNS)
        print(f"  LoRA injected: {n} layers (rank={lora_rank})")

        # DFM components
        self.pyramid_x = Pyramid4Scale(256)
        self.pyramid_y = Pyramid4Scale(256)
        self.fusion4 = SEFusion(256)
        self.fusion3 = SEFusion(256)
        self.fusion2 = SEFusion(256)
        self.fusion1 = SEFusion(256)
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=decode_channels, dropout=dropout)

        # Freeze non-LoRA backbone
        for n, p in self.backbone.named_parameters():
            if 'lora_' not in n:
                p.requires_grad = False
        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  F0'+L model: trainable={trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        backbone_out_rgb = self.backbone.forward_image(images)
        deepx = backbone_out_rgb["backbone_fpn"][-1]
        dsm_3ch = dsm.repeat(1, 3, 1, 1)
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        fx = self.pyramid_x(deepx); fy = self.pyramid_y(deepy)
        f4 = self.fusion4(fx[0], fy[0]); f3 = self.fusion3(fx[1], fy[1])
        f2 = self.fusion2(fx[2], fy[2]); f1 = self.fusion1(fx[3], fy[3])
        return self.decoder([f4, f3, f2, f1])


# Re-export FrozenSAM3DFM for convenience
from model_f0p import FrozenSAM3DFM
