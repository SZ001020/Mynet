"""P13-G from-scratch: joint training of base model + texture + nDSM roughness.

Plan7PromptMFNet (unfrozen) + MultiScaleTextureStem + NDSMRoughnessStem +
VegetationBinaryHead. Veg delta applied strictly within predicted veg mask.
Single structure_loss on refined output — no auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters
from texture_multiscale import MultiScaleTextureStem
from ndsm_roughness import NDSMRoughnessStem


class VegetationBinaryHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_ch, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Plan13GScratch(nn.Module):
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

        self.texture_stem = MultiScaleTextureStem(in_ch=3, branch_ch=48, out_ch=128)
        self.ndsm_stem = NDSMRoughnessStem(out_ch=64)
        self.veg_head = VegetationBinaryHead(5 + 128 + 64, mid_ch=128)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan13-G-scratch trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor):
        images_norm = (images - 0.5) / 0.5
        if images_norm.shape[-2:] != (self.resolution, self.resolution):
            images_norm = F.interpolate(images_norm, (self.resolution, self.resolution),
                                        mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution),
                                mode="bilinear", align_corners=False)

        patch_hw = (self.resolution // 14, self.resolution // 14)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        self.mm_state["prompt_tokens"] = self.prompt_encoder(dsm, patch_hw)

        try:
            backbone_out = self.backbone.forward_image(images_norm)
        finally:
            self.mm_state.pop("dsm_tokens", None)
            self.mm_state.pop("prompt_tokens", None)

        vit_feat = backbone_out["backbone_fpn"][-1].clone()
        feats = self.pyramid(vit_feat)
        base_logits = self.decoder(feats)

        out_size = base_logits.shape[-2:]
        texture_feat = self.texture_stem(images, out_size)
        ndsm_feat = self.ndsm_stem(dsm, out_size)

        combined = torch.cat([base_logits.float(), texture_feat, ndsm_feat], dim=1)
        delta = self.veg_head(combined)

        veg_mask = (base_logits.argmax(1) == 2) | (base_logits.argmax(1) == 3)
        delta = delta * veg_mask.unsqueeze(1).float()

        refined = base_logits.clone()
        refined[:, 2:4] = refined[:, 2:4] + delta.float()

        return refined
