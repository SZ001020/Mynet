"""
LoRA-SAM3 + UNetFormer Decoder (Plan3 Route B-hybrid)

LoRA injected into ViT attention layers (Q/K/V/out_proj) replaces VPT Adapter.
UNetFormer decoder (GLA blocks) from Route A handles multi-scale feature decoding.

Components:
- LoRA-ViT: SAM3 ViTDet with LoRA on all attention Linear layers (~8-15M params)
- FPN: frozen 3-scale feature pyramid
- UNetFormer decoder: GLA window attention (trainable)
- Optional DSM: cross-attention fusion (AdapterBlockDSM from p3r)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from lora_layers import inject_lora_into_module, count_lora_params, get_lora_params


# ── LoRA injection patterns for SAM3 components ──────────────

VISION_PATTERNS = [
    r'attn\.qkv$',       # ViTDet attention QKV (fused in SAM3)
    r'attn\.proj$',       # Attention output projection
    r'q_proj$', r'k_proj$', r'v_proj$', r'out_proj$',  # MHA separated
    r'mlp\.fc1$', r'mlp\.fc2$',  # MLP layers
]

TEXT_PATTERNS = [
    r'c_fc$', r'c_proj$',  # CLIP text MLP
]

DETR_PATTERNS = [
    r'self_attn\.', r'multihead_attn\.', r'linear', r'q_proj$', r'k_proj$',
]


class LoRASAM3UNetFormer(nn.Module):
    """Route B-hybrid: LoRA on SAM3 ViT + UNetFormer decoder.

    LoRA rank controls the adapter capacity:
    - rank=8: ~4.7M LoRA params + 6.5M decoder = ~11.2M total
    - rank=16: ~9.4M LoRA params + 6.5M decoder = ~15.9M total
    """

    def __init__(self, sam3_model, lora_rank: int = 8, lora_alpha: float = 16.0,
                 num_classes: int = 5, dropout: float = 0.1, window_size: int = 8,
                 use_dsm: bool = False, dsm_dim: int = 128):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.num_classes = num_classes

        # Freeze entire backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Inject LoRA into ViT attention layers
        vb = self.backbone.vision_backbone
        n = inject_lora_into_module(vb, rank=lora_rank, alpha=lora_alpha,
                                    target_patterns=VISION_PATTERNS)
        print(f"  LoRA injected into {n} ViT layers (rank={lora_rank}, alpha={lora_alpha})")

        lora_params = count_lora_params(vb)
        print(f"  LoRA params: {lora_params:,}")

        # Freeze language backbone (not used in hybrid mode)
        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        # DSM support (optional, cross-attention fusion like p3r v3)
        self.use_dsm = use_dsm
        if use_dsm:
            from adapter_vit import DSMEncoderDeep
            self.dsm_encoder = DSMEncoderDeep(dsm_dim=dsm_dim)
            # Also inject LoRA into DSM-compatible blocks
            self._setup_dsm_fusion(vb, dsm_dim)
            print(f"  DSM encoder: {sum(p.numel() for p in self.dsm_encoder.parameters()):,} params")

        # UNetFormer decoder
        from unetformer_decoder import UNetFormerDecoder
        self.decoder = UNetFormerDecoder(num_classes=num_classes, decode_channels=256,
                                         dropout=dropout, window_size=window_size)

        self._log_params()

    def _setup_dsm_fusion(self, vision_backbone, dsm_dim):
        """FPN-level cross-attention DSM fusion modules."""
        # Per-scale cross-attention: Q=FPN, K/V=DSM
        fpn_ch = 256
        self.dsm_fuse = nn.ModuleList([
            DSMCrossAttn(fpn_ch, dsm_dim) for _ in range(3)  # 288, 144, 72
        ])

    def _log_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if 'lora_A' in n or 'lora_B' in n)
        decoder = sum(p.numel() for p in self.decoder.parameters())
        print(f"  LoRA: {lora:,} | Decoder: {decoder:,}")
        print(f"  Total trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def extract_features(self, images, resolution=None):
        if resolution is None:
            resolution = getattr(self, 'resolution', 1008)
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)
        out = self.backbone.forward_image(images_norm)
        return [f.clone() for f in out['backbone_fpn']]

    def forward(self, images, dsm=None):
        feats = self.extract_features(images)
        if dsm is not None and self.use_dsm:
            dsm_out = self.dsm_encoder(dsm)
            dsm_scales = dsm_out['scales']
            # Cross-attention at each FPN scale: FPN features query DSM context
            feats = [self.dsm_fuse[i](feats[i], dsm_scales[i]) for i in range(3)]
        return self.decoder(feats)


class DSMCrossAttn(nn.Module):
    """Gated DSM-FPN fusion: pixel-wise learned mixing of RGB and elevation features.

    gate = sigmoid(conv([fpn, dsm_proj]))
    out = fpn * gate + dsm_proj * (1 - gate)

    Memory-efficient (no attention matrix), per-pixel adaptive fusion.
    """

    def __init__(self, fpn_ch: int = 256, dsm_ch: int = 128):
        super().__init__()
        self.dsm_proj = nn.Sequential(
            nn.Conv2d(dsm_ch, fpn_ch, 1), nn.BatchNorm2d(fpn_ch), nn.ReLU(True),
            nn.Conv2d(fpn_ch, fpn_ch, 1))
        self.gate_conv = nn.Sequential(
            nn.Conv2d(fpn_ch * 2, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(True),
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.Sigmoid())

    def forward(self, fpn_feat, dsm_feat):
        dsm = self.dsm_proj(dsm_feat)
        gate = self.gate_conv(torch.cat([fpn_feat, dsm], dim=1))
        return fpn_feat * gate + dsm * (1 - gate)
