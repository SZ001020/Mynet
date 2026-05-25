"""A3: MMAdapter full DSM attention + SimpleDecoder + Pyramid4Scale.

Same as Phase 1 full attn but with SimpleDecoder instead of MFNetDecoder.
Isolates adapter contribution from decoder contribution.
"""
from __future__ import annotations
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
BASE = "/root/Mynet"
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation")
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter")
from simple_decoder import SimpleDecoder
from mm_adapter_vit import inject_mm_adapters  # Phase 1's MMAdapter


class DSMTokenEncoder(nn.Module):
    def __init__(self, token_dim=1024, dsm_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, dsm_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
            nn.Conv2d(dsm_dim, dsm_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dsm_dim), nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(dsm_dim, token_dim, 1)

    def forward(self, dsm, spatial_hw):
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        feat = self.proj(self.stem(dsm))
        if feat.shape[-2:] != spatial_hw:
            feat = F.interpolate(feat, spatial_hw, mode="bilinear", align_corners=False)
        return feat.permute(0, 2, 3, 1).contiguous()


class Pyramid4Scale(nn.Module):
    """ViT feature → 4-scale pyramid."""
    def __init__(self, ch=256):
        super().__init__()
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ch, ch, 2, 2), nn.BatchNorm2d(ch), nn.ReLU(inplace=False))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(ch, ch, 2, 2), nn.BatchNorm2d(ch), nn.ReLU(inplace=False),
                                  nn.ConvTranspose2d(ch, ch, 2, 2), nn.BatchNorm2d(ch), nn.ReLU(inplace=False))
        self.id3 = nn.Identity()
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, feat):
        return [self.up4(feat), self.up2(feat), self.id3(feat), self.down(feat)]


class A3MMAdapterFull(nn.Module):
    """MMAdapter full DSM attention + Pyramid4Scale + SimpleDecoder."""

    def __init__(self, sam3_model, num_classes=5, bottleneck=32, dsm_dim=128):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = 1008

        # Inject MMAdapter blocks (from Phase 1's mm_adapter_vit)
        self.mm_state = inject_mm_adapters(
            self.backbone.vision_backbone, bottleneck=bottleneck,
            dsm_attn_mode="full", checkpoint_attn=True)

        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.pyramid = Pyramid4Scale(256)
        self.decoder = SimpleDecoder(in_channels=256, decode_channels=64, num_classes=num_classes)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  A3 MMAdapter+SimpleDecoder: {trainable:,} trainable / {total:,} total")

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        patch_hw = (self.resolution // 16, self.resolution // 16)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        try:
            out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)

        feat = out["backbone_fpn"][-1].clone()
        scales = self.pyramid(feat)
        logits = self.decoder(scales[-1])
        return F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
