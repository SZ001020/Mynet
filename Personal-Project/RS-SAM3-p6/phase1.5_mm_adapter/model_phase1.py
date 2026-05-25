"""Plan6 Phase 1 model: SAM3 ViTDet + in-ViT RGB/DSM MMAdapter + MFNet decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters


class Plan6MMAdapterMFNet(nn.Module):
    """Frozen SAM3 backbone with trainable in-ViT RGB/DSM MMAdapters."""

    def __init__(
        self,
        sam3_model,
        adapter_bottleneck: int = 32,
        num_classes: int = 5,
        dropout: float = 0.1,
        dsm_dim: int = 128,
        dsm_attn_mode: str = "adapter",
        checkpoint_attn: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = 1008
        self.mm_state = inject_mm_adapters(
            self.backbone.vision_backbone,
            bottleneck=adapter_bottleneck,
            dsm_attn_mode=dsm_attn_mode,
            checkpoint_attn=checkpoint_attn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.pyramid = Pyramid4Scale(256)
        self.decoder = MFNetDecoder(
            num_classes=num_classes,
            decode_channels=64,
            dropout=dropout,
        )

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan6 model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(
                images,
                (self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(
                dsm,
                (self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        patch_hw = (self.resolution // 16, self.resolution // 16)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        try:
            backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)

        vit_feat = backbone_out["backbone_fpn"][-1].clone()
        feats = self.pyramid(vit_feat)
        return self.decoder(feats)
