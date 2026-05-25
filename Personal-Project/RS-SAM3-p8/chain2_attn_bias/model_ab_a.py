"""Plan8 AB-A model: SAM3 + RGB/DSM/prompt MMAdapter w/ DSM elev attn bias + MFNet decoder."""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared modules from Plan7-A
PHASE7_DIR = "/root/Mynet/RS-SAM3-p7/phase_a_dsm_prompt"
if PHASE7_DIR not in sys.path:
    sys.path.insert(0, PHASE7_DIR)

from dsm_prompt import PromptEncoder  # noqa: E402
from mfnet_decoder import MFNetDecoder, Pyramid4Scale  # noqa: E402
from mm_adapter_vit_ab import DSMTokenEncoder, inject_mm_adapters  # noqa: E402


class Plan8ABMFNet(nn.Module):
    """Plan8 Chain 2 Phase A: DSM elevation-difference attention bias with MFNet decoder.

    Architecture = Plan7PromptMFNet + DSMAttentionBias at global blocks.
    """

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

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan8-AB model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        patch_hw = (self.resolution // 16, self.resolution // 16)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        self.mm_state["prompt_tokens"] = self.prompt_encoder(dsm, patch_hw)
        try:
            backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)
            self.mm_state.pop("prompt_tokens", None)

        vit_feat = backbone_out["backbone_fpn"][-1].clone()
        feats = self.pyramid(vit_feat)
        return self.decoder(feats)
