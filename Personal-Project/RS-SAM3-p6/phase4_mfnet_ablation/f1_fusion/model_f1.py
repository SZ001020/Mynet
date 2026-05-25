"""F1/F2/F3 unified model: Plan6/7 MMAdapter + optional LoRA + optional prompt.

F1: independent encoder + in-ViT MMAdapter + LoRA + MFNetDecoder + No prompt
F2: independent encoder + in-ViT MMAdapter + Frozen + MFNetDecoder + No prompt
F3: independent encoder + in-ViT MMAdapter + Frozen + MFNetDecoder + edge/slope prompt

All share the same core architecture; flags control LoRA and prompt inclusion.
"""

from __future__ import annotations

import sys, os

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_A)
sys.path.insert(0, SHARED)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_prompt import PromptEncoder
from mfnet_decoder import MFNetDecoder, Pyramid4Scale
from mm_adapter_vit import DSMTokenEncoder, inject_mm_adapters
from lora_layers import inject_lora_into_module


class Phase4AdapterModel(nn.Module):
    """Unified F1/F2/F3 model.

    F1: use_lora=True,  use_prompt=False → In-ViT MMAdapter + LoRA
    F2: use_lora=False, use_prompt=False → In-ViT MMAdapter + Frozen (=Plan6 Phase1)
    F3: use_lora=False, use_prompt=True  → In-ViT MMAdapter + Frozen + prompt (=Plan7-A)
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
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        use_prompt: bool = False,
    ):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.resolution = resolution
        self.use_prompt = use_prompt

        # ── Step 1: In-ViT MMAdapter (must come before LoRA!) ──
        self.mm_state = inject_mm_adapters(
            self.backbone.vision_backbone,
            bottleneck=adapter_bottleneck,
            dsm_attn_mode=dsm_attn_mode,
            checkpoint_attn=checkpoint_attn,
        )
        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)

        # ── Step 2: Optional LoRA (on already-adapter-injected blocks) ──
        self.use_lora = use_lora
        if use_lora:
            VISION_PATTERNS = [
                r'attn\.qkv$', r'attn\.proj$', r'q_proj$', r'k_proj$', r'v_proj$', r'out_proj$',
                r'mlp\.fc1$', r'mlp\.fc2$',
            ]
            trunk = self.backbone.vision_backbone.trunk
            n = inject_lora_into_module(trunk, rank=lora_rank, alpha=lora_alpha, target_patterns=VISION_PATTERNS)
            print(f"  LoRA injected: {n} layers (rank={lora_rank})")

        # ── Optional Prompt ──
        if use_prompt:
            self.prompt_encoder = PromptEncoder(token_dim=1024, prompt_dim=prompt_dim)

        # ── Shared decoder ──
        self.pyramid = Pyramid4Scale(256)
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64, dropout=dropout)

        # ── Freeze logic ──
        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters():
                p.requires_grad = False

        if not use_lora:
            # Fully frozen backbone (only adapter + decoder trainable)
            for n, p in self.backbone.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        flags = []
        if use_lora: flags.append("LoRA")
        if use_prompt: flags.append("prompt")
        tag = "+".join(flags) if flags else "frozen-only"
        print(f"  Phase4 model ({tag}): trainable={trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

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
        if self.use_prompt:
            self.mm_state["prompt_tokens"] = self.prompt_encoder(dsm, patch_hw)
        else:
            # MMAdapterPromptBlock expects prompt_tokens; supply zeros when disabled
            self.mm_state["prompt_tokens"] = torch.zeros(
                images.shape[0], patch_hw[0], patch_hw[1], 1024,
                device=images.device, dtype=torch.float32
            )
        try:
            backbone_out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)
            self.mm_state.pop("prompt_tokens", None)

        vit_feat = backbone_out["backbone_fpn"][-1].clone()
        feats = self.pyramid(vit_feat)
        return self.decoder(feats)
