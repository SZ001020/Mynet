"""A1: A0 + DSM late fusion via SEFusion + 4× upsample decoder.

DSM → CNN encoder → 4-scale SEFusion after Pyramid4Scale → SimpleDecoder.
This measures: DSM's value when fused OUTSIDE the ViT (Plan3-style).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from mfnet_decoder import SEFusion, Pyramid4Scale


class DSMLateFusion(nn.Module):
    def __init__(self, sam3_model, num_classes: int = 5, dsm_dim: int = 128):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = 1008
        # DSM encoder (same as Phase 1)
        self.dsm_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, dsm_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
            nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
        )
        self.pyramid = Pyramid4Scale(256)
        self.f1 = SEFusion(256); self.f2 = SEFusion(256)
        self.f3 = SEFusion(256); self.f4 = SEFusion(256)
        self.dsm_proj = nn.ModuleList([nn.Conv2d(dsm_dim, 256, 1) for _ in range(4)])

        # Simple upsample decoder (not MFNetDecoder)
        decode_ch = 64
        self.pre_conv = nn.Conv2d(256, decode_ch, 1)
        self.up_blocks = nn.ModuleList([
            self._up_block(decode_ch) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(decode_ch, decode_ch, 3, padding=1), nn.BatchNorm2d(decode_ch), nn.ReLU(inplace=False),
            nn.Conv2d(decode_ch, num_classes, 1),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  A1 late fusion: {trainable:,} trainable / {total:,} total")

    def _up_block(self, ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=False),
        )

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        out = self.backbone.forward_image(images)
        vit_feat = out["backbone_fpn"][-1]
        rgb_scales = self.pyramid(vit_feat)

        dsm_feat = self.dsm_encoder(dsm)
        if dsm_feat.shape[-2:] != vit_feat.shape[-2:]:
            dsm_feat = F.interpolate(dsm_feat, vit_feat.shape[-2:], mode="bilinear", align_corners=False)
        # Resize DSM to match each RGB pyramid scale
        dsm_scales = [F.interpolate(dsm_feat, s.shape[-2:], mode="bilinear", align_corners=False)
                      for s in rgb_scales]

        feats = [getattr(self, f"f{i+1}")(rgb_scales[i], self.dsm_proj[i](dsm_scales[i]))
                 for i in range(4)]

        x = self.pre_conv(feats[-1])
        for up in self.up_blocks:
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        logits = self.head(x)
        return F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
