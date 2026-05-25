"""F0: MFNet-style architecture on SAM3 backbone.

Shared encoder (SAM3 ViTDet) → deepx/deepy → SEFusion → Pyramid4Scale → MFNetDecoder.
LoRA injected on ViTDet attention layers (rank=8, same as MFNet paper).

Key: SAM3 ViTDet outputs all backbone_fpn features at the same resolution (1/16).
We fuse the LAST features from both modalities via SEFusion, then use Pyramid4Scale
to create the 4-scale pyramid for the decoder. This is functionally equivalent to
MFNet's late fusion approach, adapted to SAM3's feature structure.
"""

from __future__ import annotations

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
sys.path.insert(0, SE)

SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
sys.path.insert(0, SHARED)

from se_fusion import SEFusion
from lora_layers import inject_lora_into_module


# ═══ F0 Model ═══

class MFNetSAM3(nn.Module):
    """MFNet exact architecture on SAM3 backbone.

    RGB and DSM share the same SAM3 ViTDet encoder (run twice).
    LoRA on attention QKV and MLP layers (rank=8, alpha=16) replaces
    MFNet's SAM1 LoRA.
    """

    def __init__(
        self,
        sam3_model,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        num_classes: int = 5,
        decode_channels: int = 64,
        dropout: float = 0.1,
        encoder_channels=(256, 256, 256, 256),
        resolution: int = 1008,
    ):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution
        self.num_classes = num_classes

        # ── LoRA injection on SAM3 ViTDet ──
        VISION_PATTERNS = [
            r'attn\.qkv$',
            r'attn\.proj$',
            r'q_proj$', r'k_proj$', r'v_proj$', r'out_proj$',
            r'mlp\.fc1$', r'mlp\.fc2$',
        ]
        trunk = self.backbone.vision_backbone.trunk
        n_injected = inject_lora_into_module(trunk, rank=lora_rank, alpha=lora_alpha,
                                             target_patterns=VISION_PATTERNS)
        print(f"  LoRA injected: {n_injected} layers (rank={lora_rank})")

        # ── SEFusion: late fusion of RGB and DSM features ──
        self.fusion = SEFusion(256)

        # ── Pyramid + MFNetDecoder ──
        PHASE1_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
        sys.path.insert(0, PHASE1_DIR)
        from mfnet_decoder import MFNetDecoder, Pyramid4Scale

        self.pyramid = Pyramid4Scale(256)
        self.decoder = MFNetDecoder(
            num_classes=num_classes, decode_channels=decode_channels, dropout=dropout)

        # Freeze non-LoRA backbone params
        for n, p in self.backbone.named_parameters():
            if 'lora_' not in n:
                p.requires_grad = False

        # Language backbone stays frozen
        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  F0 model: trainable={trainable:,} / total={total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution),
                                   mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution),
                                mode="bilinear", align_corners=False)

        # ── Shared encoder: run SAM3 backbone twice (RGB + DSM-as-RGB) ──
        backbone_out_rgb = self.backbone.forward_image(images)
        deepx = backbone_out_rgb["backbone_fpn"][-1]  # Last scale, B×256×h×w

        dsm_3ch = dsm.repeat(1, 3, 1, 1)
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        # ── Late SEFusion + Pyramid4Scale + MFNetDecoder ──
        fused = self.fusion(deepx, deepy)
        feats = self.pyramid(fused)
        return self.decoder(feats)
