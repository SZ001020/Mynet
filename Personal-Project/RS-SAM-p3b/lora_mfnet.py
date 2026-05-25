"""
MFNet-style deep DSM fusion: DSM goes through same SAM3 ViT as RGB (shared weights).

Key MFNet design principles applied to our SAM3 + LoRA setup:
1. DSM → 3-channel (gradient-x, gradient-y, original) → SAM3 ViT (shared LoRA weights)
2. Both modalities produce FPN features through the SAME encoder
3. SEFusion: per-scale SE channel attention on each modality → sum
4. 4-scale pyramid (1/4, 1/8, 1/16, 1/32) via FPN expansion
5. UNetFormer decoder (4-scale)

This replaces the weak 3-layer CNN DSM encoder with SAM3's full 32-block ViT.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from lora_layers import inject_lora_into_module, count_lora_params
from unetformer_decoder import (Conv, ConvBN, ConvBNReLU, SeparableConvBN,
                                 GlobalLocalAttention, GLABlock, Mlp, DropPath)


# ── DSM to 3-channel conversion ─────────────────────────────

def dsm_to_3ch(dsm_1ch):
    """Convert single-channel DSM to 3-channel for SAM3 ViT input.

    MFNet simply repeats DSM 3 times. We do gradient-x + gradient-y + original
    to provide richer geometric cues (slope in x/y directions + absolute height).
    """
    if dsm_1ch.dim() == 3:
        dsm_1ch = dsm_1ch.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

    B, _, H, W = dsm_1ch.shape
    # Normalize to [0, 1] per sample
    dsm_flat = dsm_1ch.view(B, -1)
    d_min = dsm_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    d_max = dsm_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    d_norm = (dsm_1ch - d_min) / (d_max - d_min + 1e-8)

    # Sobel gradients for slope information
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                           device=dsm_1ch.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(-2, -1)
    grad_x = F.conv2d(F.pad(d_norm, (1,1,1,1), mode='reflect'), sobel_x)
    grad_y = F.conv2d(F.pad(d_norm, (1,1,1,1), mode='reflect'), sobel_y)
    grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-8)
    grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min() + 1e-8)

    return torch.cat([grad_x, grad_y, d_norm], dim=1)  # (B, 3, H, W)


# ── SEFusion (from MFNet) ──────────────────────────────────

class SqueezeAndExcitation(nn.Module):
    """SE channel attention block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class SEFusion(nn.Module):
    """SE attention per modality → element-wise sum (MFNet's fusion)."""
    def __init__(self, channels):
        super().__init__()
        self.se_rgb = SqueezeAndExcitation(channels)
        self.se_dsm = SqueezeAndExcitation(channels)

    def forward(self, rgb_feat, dsm_feat):
        return self.se_rgb(rgb_feat) + self.se_dsm(dsm_feat)


# ── MFNet-style Decoder (4-scale) ───────────────────────────

class WFSingle4(nn.Module):
    """Weighted fusion for 4-scale decoder (1 channel input)."""
    def __init__(self, in_ch, decode_ch):
        super().__init__()
        self.pre = Conv(in_ch, decode_ch, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, x, skip):
        skip = self.pre(skip)
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w = torch.relu(self.weights)
        return (w[0] * x + w[1] * skip) / (w[0] + w[1] + 1e-8)


class UNetFormerDecoder4(nn.Module):
    """UNetFormer decoder for 4-scale features (1/4, 1/8, 1/16, 1/32).

    Matches MFNet's decoder structure with 4 input scales.
    """

    def __init__(self, num_classes=5, decode_channels=256, dropout=0.1, window_size=8):
        super().__init__()
        # encoder_channels: (res1@1/4, res2@1/8, res3@1/16, res4@1/32)
        # Deepest: 1/32 (feature map ~36² at 1008 input — use small window)
        self.pre_conv = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.b4 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=min(window_size, 4))

        # 1/16
        self.p3 = WFSingle4(decode_channels, decode_channels)
        self.b3 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # 1/8
        self.p2 = WFSingle4(decode_channels, decode_channels)
        self.b2 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # 1/4
        self.p1 = WFSingle4(decode_channels, decode_channels)
        self.b1 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # Head
        self.head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(dropout, inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            Conv(decode_channels, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        """feats: [f1@1/4, f2@1/8, f3@1/16, f4@1/32]"""
        f1, f2, f3, f4 = feats

        # Bottleneck: 1/32
        x = self.b4(self.pre_conv(f4))

        # 1/32 → 1/16
        x = F.interpolate(x, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.p3(x, f3)
        x = self.b3(x)

        # 1/16 → 1/8
        x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.p2(x, f2)
        x = self.b2(x)

        # 1/8 → 1/4
        x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.p1(x, f1)
        x = self.b1(x)

        return self.head(x)


# ── LoRA-SAM3 MFNet-style model ─────────────────────────────

VISION_PATTERNS = [r'attn\.qkv$', r'attn\.proj$', r'q_proj$', r'k_proj$', r'v_proj$',
                   r'out_proj$', r'mlp\.fc1$', r'mlp\.fc2$']


class LoRASAM3MFNet(nn.Module):
    """MFNet-inspired dual-branch SAM3 with deep DSM fusion.

    RGB and DSM both go through the SAME SAM3 ViT (shared LoRA weights),
    producing FPN features at 3 scales. Features are expanded to 4 scales
    and fused via SEFusion, then decoded by UNetFormer (4-scale).
    """

    def __init__(self, sam3_model, lora_rank=8, lora_alpha=16.0,
                 num_classes=5, dropout=0.1, window_size=8):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.num_classes = num_classes

        # Freeze backbone, inject LoRA
        for p in self.backbone.parameters():
            p.requires_grad = False

        vb = self.backbone.vision_backbone
        n = inject_lora_into_module(vb, rank=lora_rank, alpha=lora_alpha,
                                    target_patterns=VISION_PATTERNS)
        print(f"  LoRA injected into {n} ViT layers (rank={lora_rank})")
        print(f"  LoRA params: {count_lora_params(vb):,}")

        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        # 4-scale pyramid: expand 3 FPN scales → 4 MFNet scales
        # FPN gives ~1/3.5, 1/7, 1/14. We map to MFNet's 1/4, 1/8, 1/16, 1/32.
        self.pyramid_up2 = nn.Sequential(  # 1/14 → 1/4 (upsample 3.5x)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBN(256, 256, kernel_size=3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBN(256, 256, kernel_size=3),
        )
        self.pyramid_up1 = nn.Sequential(  # 1/14 → 1/8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBN(256, 256, kernel_size=3),
        )
        self.pyramid_base = nn.Identity()  # 1/14 ≈ 1/16
        self.pyramid_down = nn.Sequential(  # 1/14 → 1/32
            nn.AvgPool2d(2, 2),
            ConvBN(256, 256, kernel_size=1),
        )

        # SEFusion per scale
        self.fusion1 = SEFusion(256)  # 1/4
        self.fusion2 = SEFusion(256)  # 1/8
        self.fusion3 = SEFusion(256)  # 1/16
        self.fusion4 = SEFusion(256)  # 1/32

        # 4-scale UNetFormer decoder
        self.decoder = UNetFormerDecoder4(num_classes=num_classes,
                                          decode_channels=256, dropout=dropout,
                                          window_size=window_size)
        self._log_params()

    def _log_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if 'lora_A' in n or 'lora_B' in n)
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        print(f"  LoRA: {lora:,} | Decoder: {decoder_p:,}")
        print(f"  Total trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def _extract_one(self, images_3ch, resolution=1008):
        """Forward one modality through SAM3 ViT, return FPN features."""
        images_norm = (images_3ch - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)
        out = self.backbone.forward_image(images_norm)
        return [f.clone() for f in out['backbone_fpn']]  # 3 scales

    def _fpn_to_4scales(self, fpn_feats):
        """Convert 3 FPN scales to 4 MFNet scales."""
        f0, f1, f2 = fpn_feats  # 1/3.5, 1/7, 1/14
        s1 = self.pyramid_up2(f2)   # 1/4
        s2 = self.pyramid_up1(f2)   # 1/8
        s3 = self.pyramid_base(f2)  # 1/16
        s4 = self.pyramid_down(f2)  # 1/32
        return [s1, s2, s3, s4]

    def forward(self, images_rgb, dsm_1ch=None):
        """Forward RGB and optional DSM through shared SAM3 ViT."""
        # RGB branch
        fpn_rgb = self._extract_one(images_rgb)
        scales_rgb = self._fpn_to_4scales(fpn_rgb)

        if dsm_1ch is not None:
            # DSM → 3-channel → SAM3 ViT (shared weights, separate forward)
            dsm_3ch = dsm_to_3ch(dsm_1ch)
            fpn_dsm = self._extract_one(dsm_3ch)
            scales_dsm = self._fpn_to_4scales(fpn_dsm)

            # SEFusion at each scale
            feats = [
                self.fusion1(scales_rgb[0], scales_dsm[0]),
                self.fusion2(scales_rgb[1], scales_dsm[1]),
                self.fusion3(scales_rgb[2], scales_dsm[2]),
                self.fusion4(scales_rgb[3], scales_dsm[3]),
            ]
        else:
            feats = scales_rgb

        return self.decoder(feats)
