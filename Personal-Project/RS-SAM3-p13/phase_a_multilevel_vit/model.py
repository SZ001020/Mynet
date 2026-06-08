"""Plan13-A model: SAM3 + multi-level ViT feature pyramid + MMAdapter + MFNet decoder.

Key change from Plan7-A: instead of using only block-31 features for all decoder
scales (via Pyramid4Scale), this extracts features from 4 ViT depths (blocks 7, 15,
23, 31) and routes each to the corresponding decoder scale. Early blocks retain
local texture information that is critical for tree/grass discrimination.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import MFNetDecoder
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters


class MultiLevelPyramid(nn.Module):
    """Build 4-scale pyramid from 4 ViT intermediate features at different depths.

    Unlike Pyramid4Scale (which creates all scales from a single deep feature via
    ConvTranspose), this routes each ViT depth to its corresponding decoder scale:
      block_7  (texture-rich) -> 1/4  (288^2)
      block_15                 -> 1/8  (144^2)
      block_23                 -> 1/16 (72^2)
      block_31 (semantic)      -> 1/32 (36^2)
    """

    def __init__(self, vit_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.proj7 = nn.Conv2d(vit_dim, out_dim, 1)
        self.proj15 = nn.Conv2d(vit_dim, out_dim, 1)
        self.proj23 = nn.Conv2d(vit_dim, out_dim, 1)
        self.proj31 = nn.Conv2d(vit_dim, out_dim, 1)

    def forward(self, vit_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """vit_features: [feat_7, feat_15, feat_23, feat_31], each [B,1024,72,72]."""
        f7, f15, f23, f31 = vit_features

        s1 = F.interpolate(self.proj7(f7), scale_factor=4.0, mode="bilinear", align_corners=False)
        s2 = F.interpolate(self.proj15(f15), scale_factor=2.0, mode="bilinear", align_corners=False)
        s3 = self.proj23(f23)
        s4 = F.max_pool2d(self.proj31(f31), kernel_size=2, stride=2)

        return [s1, s2, s3, s4]


class Plan13MultiLevelMFNet(nn.Module):
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

        # Enable multi-level ViT feature extraction
        trunk = self.backbone.vision_backbone.trunk
        trunk.return_interm_layers = True

        self.mm_state = inject_mm_adapters(
            self.backbone.vision_backbone,
            bottleneck=adapter_bottleneck,
            dsm_attn_mode=dsm_attn_mode,
            checkpoint_attn=checkpoint_attn,
        )

        # Capture ViT intermediate features via forward hook
        self._vit_features: list[torch.Tensor] | None = None

        def _capture_vit(module, _input, output):
            self._vit_features = list(output)

        trunk.register_forward_hook(_capture_vit)

        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.prompt_encoder = PromptEncoder(token_dim=1024, prompt_dim=prompt_dim)

        # Multi-level pyramid replaces Pyramid4Scale
        self.pyramid = MultiLevelPyramid(vit_dim=1024, out_dim=256)

        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64, dropout=dropout)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan13-A model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
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

        self._vit_features = None
        try:
            _backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)
            self.mm_state.pop("prompt_tokens", None)

        if self._vit_features is None or len(self._vit_features) != 4:
            raise RuntimeError(
                f"Expected 4 ViT intermediate features (blocks 7,15,23,31), "
                f"got {self._vit_features}. Check return_interm_layers=True."
            )

        feats = self.pyramid(self._vit_features)
        return self.decoder(feats)
