"""P13-C model: Plan7PromptMFNet + RGB texture stem + vegetation refinement head.

Adds a lightweight CNN (TextureStem) that extracts fine texture features directly
from raw RGB pixels at 1/4 resolution, bypassing SAM3 ViT's 14x14 patch bottleneck.
A vegetation refinement head fuses texture + semantic features to produce tree/grass
delta logits, applied only within vegetation regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters
from texture_stem import TextureStem


class VegetationRefinementHead(nn.Module):
    """Fuses texture + semantic features to produce tree/grass delta logits.

    Takes pyramid features (before decoder) and texture features, outputs
    tree/grass delta logits that are added to the main decoder output.
    """

    def __init__(self, in_channels: int = 512, mid_channels: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, 2, 1),  # tree/grass delta logits
        )

    def forward(self, semantic_feat, texture_feat):
        """Fuse pyramid semantic feature with raw RGB texture feature.

        Args:
            semantic_feat: [B, 256, H/4', W/4'] from pyramid (ViT-derived)
            texture_feat: [B, 256, H/4', W/4'] from TextureStem (raw RGB)

        Returns:
            delta: [B, 2, H/4', W/4'] tree/grass delta logits
        """
        fused = torch.cat([semantic_feat, texture_feat], dim=1)
        return self.fusion(fused)


class Plan13CTextureMFNet(nn.Module):
    def __init__(
        self,
        sam3_model,
        adapter_bottleneck: int = 32,
        num_classes: int = 5,
        dropout: float = 0.1,
        dsm_dim: int = 128,
        prompt_dim: int = 128,
        dsm_attn_mode: str = "full",
        checkpoint_attn: bool = False,
        resolution: int = 1008,
    ):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution
        self.mm_state = inject_mm_adapters(
            self.backbone.vision_backbone,
            bottleneck=adapter_bottleneck,
            dsm_attn_mode=dsm_attn_mode,
            checkpoint_attn=checkpoint_attn,
        )
        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.prompt_encoder = PromptEncoder(token_dim=1024, prompt_dim=prompt_dim)
        self.pyramid = Pyramid4Scale(256)
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64, dropout=dropout)

        self.texture_stem = TextureStem(in_ch=3, out_ch=256)
        self.veg_refine_head = VegetationRefinementHead(in_channels=512, mid_channels=128)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan13-C model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        patch_hw = (self.resolution // 14, self.resolution // 14)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        self.mm_state["prompt_tokens"] = self.prompt_encoder(dsm, patch_hw)

        try:
            backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)
            self.mm_state.pop("prompt_tokens", None)

        vit_feat = backbone_out["backbone_fpn"][-1].clone()
        feats = self.pyramid(vit_feat)  # [1/4=288², 1/8=144², 1/16=72², 1/32=36²]
        decoder_out = self.decoder(feats)  # [B, 5, 288, 288]

        # Texture branch — match pyramid 1/4 spatial size
        texture_feat = self.texture_stem(images)  # [B, 256, 252, 252]
        if texture_feat.shape[-2:] != feats[0].shape[-2:]:
            texture_feat = F.interpolate(texture_feat, feats[0].shape[-2:], mode="bilinear", align_corners=False)

        # Vegetation refinement: fuse pyramid 1/4 (ViT) + texture (CNN)
        veg_delta = self.veg_refine_head(feats[0], texture_feat)  # [B, 2, 288, 288]

        # Merge delta into decoder output: grass=ch2, tree=ch3
        refined = decoder_out.clone()
        refined[:, 2:4] = refined[:, 2:4] + veg_delta

        return refined, veg_delta
