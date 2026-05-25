"""F0': Frozen SAM3 + 4xSEFusion (DFM) + MFNetDecoder.

Strict MFNet "Without Adapter" equivalent on SAM3.
Pure frozen backbone, no LoRA, no adapter, no prompt.
Two Pyramid4Scale modules + 4xSEFusion (MFNet-style DFM) + MFNetDecoder.
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


class FrozenSAM3DFM(nn.Module):
    """Frozen SAM3 ViTDet + 4xSEFusion (DFM) + MFNetDecoder.

    Matches MFNet's "Without Adapter" architecture:
    - SAM3 ViTDet: fully frozen, no LoRA, no adapter
    - RGB and DSM each pass through shared backbone separately
    - Two Pyramid4Scale modules build 4-scale representations
    - 4xSEFusion at each scale (MFNet-style DFM)
    - MFNetDecoder produces final logits
    """

    def __init__(self, sam3_model, num_classes=5, decode_channels=64,
                 dropout=0.1, resolution=1008):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution

        # Two pyramids (one per modality, matching MFNet's DFM)
        self.pyramid_x = Pyramid4Scale(256)
        self.pyramid_y = Pyramid4Scale(256)

        # 4xSEFusion at scales 1/4, 1/8, 1/16, 1/32
        self.fusion4 = SEFusion(256)  # 1/4
        self.fusion3 = SEFusion(256)  # 1/8
        self.fusion2 = SEFusion(256)  # 1/16
        self.fusion1 = SEFusion(256)  # 1/32

        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=decode_channels,
                                     dropout=dropout)

        # Fully freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  F0' model: trainable={trainable:,} / {total:,} ({trainable/ total*100:.2f}%)")

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution),
                                   mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution),
                                mode="bilinear", align_corners=False)

        # Shared encoder: run SAM3 twice
        backbone_out_rgb = self.backbone.forward_image(images)
        deepx = backbone_out_rgb["backbone_fpn"][-1]

        dsm_3ch = dsm.repeat(1, 3, 1, 1)
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        # DFM: two pyramids → 4×SEFusion
        fx = self.pyramid_x(deepx)  # [1/4, 1/8, 1/16, 1/32]
        fy = self.pyramid_y(deepy)

        # 4xSEFusion (one per scale, MFNet-style)
        f4 = self.fusion4(fx[0], fy[0])  # 1/4
        f3 = self.fusion3(fx[1], fy[1])  # 1/8
        f2 = self.fusion2(fx[2], fy[2])  # 1/16
        f1 = self.fusion1(fx[3], fy[3])  # 1/32

        return self.decoder([f4, f3, f2, f1])
