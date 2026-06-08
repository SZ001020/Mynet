"""P13-F model: P13-E + per-class vegetation gates.

Key change from P13-E: replace single veg confidence gate with two per-class
gates (tree_gate, grass_gate). Each class independently learns when to suppress
its delta on non-vegetation surfaces. This fixes P13-E's problem where the
shared gate suppressed tree predictions along with grass on building textures.

Output: [tree_delta, grass_delta, tree_gate_logit, grass_gate_logit]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters
from texture_stem import TextureStem


class PerClassGatedRefinementHead(nn.Module):
    """Fuses texture + semantic features, outputs per-class gated tree/grass delta.

    Outputs 4 channels: [tree_delta, grass_delta, tree_gate, grass_gate].
    Each gate (sigmoid) independently modulates its class delta, allowing
    tree and grass to learn different suppression patterns on non-veg surfaces.
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
            nn.Conv2d(mid_channels // 2, 4, 1),  # tree_d, grass_d, tree_g, grass_g
        )

    def forward(self, semantic_feat, texture_feat):
        """Returns (gated_delta, gate_logits).

        Args:
            semantic_feat: [B, 256, H, W] from pyramid 1/4 scale
            texture_feat: [B, 256, H, W] from TextureStem

        Returns:
            delta_gated: [B, 2, H, W] gated tree/grass delta
            gate_logits: [B, 2, H, W] raw gate logits [tree_g, grass_g]
        """
        fused = torch.cat([semantic_feat, texture_feat], dim=1)
        out = self.fusion(fused)  # [B, 4, H, W]
        raw_delta = out[:, :2]     # tree, grass delta
        gate_logits = out[:, 2:4]  # tree gate, grass gate
        gate = torch.sigmoid(gate_logits)
        return raw_delta * gate, gate_logits


class Plan13FPerClassMFNet(nn.Module):
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
        self.veg_refine_head = PerClassGatedRefinementHead(in_channels=512, mid_channels=128)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan13-F model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

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
        feats = self.pyramid(vit_feat)
        decoder_out = self.decoder(feats)

        texture_feat = self.texture_stem(images)
        if texture_feat.shape[-2:] != feats[0].shape[-2:]:
            texture_feat = F.interpolate(texture_feat, feats[0].shape[-2:], mode="bilinear", align_corners=False)

        veg_delta, gate_logits = self.veg_refine_head(feats[0], texture_feat)

        refined = decoder_out.clone()
        refined[:, 2:4] = refined[:, 2:4] + veg_delta  # grass=ch2, tree=ch3

        return refined, veg_delta, gate_logits
