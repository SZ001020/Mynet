"""A0 anchor: SAM3 raw frozen ViTDet + Conv2d head. No adapter, no DSM, no Pyramid4Scale.

This is the minimal configuration — SAM3 features directly classified.
Serves as the anchor point for Chain 1 (adapter) and Chain 5 (decoder).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class A0Anchor(nn.Module):
    def __init__(self, sam3_model, num_classes: int = 5):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = 1008
        self.head = nn.Conv2d(256, num_classes, 1)

        for param in self.parameters():
            param.requires_grad = True
        # Freeze ViT backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Only head is trainable
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  A0 anchor: {trainable:,} trainable / {total:,} total")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor = None) -> torch.Tensor:
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution),
                                   mode="bilinear", align_corners=False)
        out = self.backbone.forward_image(images)
        feat = out["backbone_fpn"][-1]  # (B, 256, 72, 72)
        logits = self.head(feat)
        return F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
