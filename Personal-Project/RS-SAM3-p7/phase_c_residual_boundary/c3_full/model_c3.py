"""Plan7-C3 model: Plan7-A + SAM-HQ residual correction.

Adds GlobalLocalFusion + ResidualCorrectionHead to Plan7PromptMFNet.
Uses forward hooks to extract intermediate features without modifying
SAM3 ViTDet or MFNetDecoder internals.

Key: logits_final = decoder_logits + residual_logits
"""

from __future__ import annotations

import sys
import os

BASE = "/root/Mynet"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
sys.path.insert(0, PHASE_A)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import Pyramid4Scale
from mfnet_decoder_c3 import MFNetDecoderC3
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters
from hq_fusion import GlobalLocalFusion
from residual_head import ResidualCorrectionHead


class Plan7C3MFNet(nn.Module):
    """Plan7-A + SAM-HQ residual boundary correction.

    Inherits the full Plan7-A architecture (RGB/DSM/prompt MMAdapter + MFNetDecoder)
    and adds a lightweight GlobalLocalFusion + ResidualCorrectionHead.
    All original weights frozen; only fusion + residual head trained.
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
        self.decoder = MFNetDecoderC3(num_classes=num_classes, decode_channels=64, dropout=dropout)

        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters():
                param.requires_grad = False

        # ── C3 new components ──
        self.hq_fusion = GlobalLocalFusion(early_dim=1024, late_dim=256, decoder_dim=64, out_dim=64)
        self.residual_head = ResidualCorrectionHead(in_channels=64, num_classes=num_classes, dropout=dropout)

        # Forward hooks for block 6 feature capture
        self._block6_feat = None
        self._hook_handles = []

        self._register_hooks()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Plan7-C3 model trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    def _register_hooks(self):
        # Block 6 hook: capture output of MMAdapterPromptBlock at index 6
        # (last pure window-attn block in SAM3 ViTDet, global attn at [7,15,23,31])
        # vision_backbone is Sam3DualViTDetNeck → blocks in vision_backbone.trunk
        trunk = self.backbone.vision_backbone.trunk
        blk6 = trunk.blocks[6]
        h1 = blk6.register_forward_hook(self._hook_block6)
        self._hook_handles.append(h1)
        # b3 feature is exposed via MFNetDecoderC3.b3_output (no hook needed)

    def _hook_block6(self, module, input, output):
        # ViTDet forward: output is BHWC tensor (or tuple with first element being tensor)
        if isinstance(output, tuple):
            self._block6_feat = output[0]
        else:
            self._block6_feat = output

    def forward(self, images: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        # ── Plan7-A forward (same as base) ──
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

        # ── Extract intermediate features ──
        # Block 6: BHWC → BCHW (1024-dim raw ViT feature, local boundary details)
        block6_feat = self._block6_feat
        if block6_feat is not None and block6_feat.ndim == 4 and block6_feat.shape[-1] > block6_feat.shape[1]:
            block6_feat = block6_feat.permute(0, 3, 1, 2).contiguous()

        # Block 31 (late global): from backbone_fpn[-1], already BCHW × 256
        # This is the same feature the decoder pyramid uses
        block31_feat = backbone_out["backbone_fpn"][-1].clone()

        # ── Main decoder forward (MFNetDecoderC3 stores b3 output as .b3_output) ──
        feats = self.pyramid(block31_feat)
        logits_main = self.decoder(feats)
        b3_feat = self.decoder.b3_output

        # ── C3 residual correction ──
        if not hasattr(self, "_c3_warned"):
            self._c3_warned = True
            print(f"  [C3 debug] block6: {block6_feat is not None}, b3: {b3_feat is not None}, "
                  f"block6_shape: {block6_feat.shape if block6_feat is not None else 'N/A'}, "
                  f"b3_shape: {b3_feat.shape if b3_feat is not None else 'N/A'}")
        if block6_feat is not None and b3_feat is not None:
            hq_features = self.hq_fusion(block6_feat, block31_feat, b3_feat)
            # Residual head output at 256×256, need to match decoder output resolution
            logits_residual = self.residual_head(hq_features)
            if logits_residual.shape[-2:] != logits_main.shape[-2:]:
                logits_residual = F.interpolate(
                    logits_residual, logits_main.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            return logits_main + logits_residual
        else:
            # Fallback if hooks didn't fire (e.g., first call before hooks registered)
            return logits_main

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
