"""
Plan3 Route A: Adapter-SAM3-UNet 完整模型

Adapter-ViT (frozen backbone + trainable adapters)
  → FPN (frozen, 3 scales)
  → UNet Decoder (trainable, 3-level)
  → 5-channel per-class logits

训练时 per-class binary loss（structure loss），评估时 per-class Dice/IoU。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter_vit import inject_adapters


class ConvBlock(nn.Module):
    """Double conv: Conv→BN→ReLU→(Dropout)→Conv→BN→ReLU→(Dropout)"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    """3-level UNet decoder with skip connections from FPN features.

    FPN outputs: [288²@256, 144²@256, 72²@256]
    Decoder: 72→144→288→576 with ConvBlocks at each level.
    dropout=0.1 for Vaihingen (trained with regularization), 0.0 for old Potsdam ckpt.
    """

    def __init__(self, num_classes: int = 5, fpn_channels: int = 256, dropout: float = 0.1):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)
        self.conv2 = ConvBlock(fpn_channels * 2, fpn_channels, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)
        self.conv1 = ConvBlock(fpn_channels * 2, fpn_channels, dropout=dropout)

        head_layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fpn_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout2d(dropout))
        head_layers += [
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout2d(dropout))
        head_layers.append(nn.Conv2d(64, num_classes, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, feats: list) -> torch.Tensor:
        """feats: [f0@288², f1@144², f2@72²] from FPN, float32"""
        f0, f1, f2 = [f.float() for f in feats]

        # 72 → 144
        x = self.up2(f2)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, f1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, f1], dim=1))

        # 144 → 288
        x = self.up1(x)
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, f0.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, f0], dim=1))

        # 288 → 576 → num_classes
        return self.head(x)


class AdapterSAM3UNet(nn.Module):
    """Complete Plan3 Route A model.

    Components:
    - SAM3 vision_backbone + adapters (trainable prompt_learn, frozen blocks)
    - FPN (frozen)
    - UNet decoder (trainable)
    """

    def __init__(self, sam3_model, adapter_bottleneck: int = 32, num_classes: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.num_classes = num_classes

        # Inject adapters into vision_backbone
        inject_adapters(self.backbone.vision_backbone, bottleneck=adapter_bottleneck)

        # Freeze FPN (convs) and position encoding
        for n, p in self.backbone.vision_backbone.named_parameters():
            if 'prompt_learn' not in n:
                p.requires_grad = False

        # Freeze language backbone entirely (not used in Route A)
        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        # UNet decoder (trainable)
        self.decoder = UNetDecoder(num_classes=num_classes, dropout=dropout)

        self._log_params()

    def _log_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        adapter_params = sum(
            p.numel() for n, p in self.backbone.vision_backbone.named_parameters()
            if 'prompt_learn' in n
        )
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  Adapter params: {adapter_params:,}")
        print(f"  Decoder params: {decoder_params:,}")
        print(f"  Total trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def extract_features(self, images: torch.Tensor, resolution: int = None) -> list:
        """Extract FPN features from the adapted ViT backbone.

        Gradients flow through adapter prompt_learn params, not frozen blocks.
        images: (B, 3, H, W), range [0, 1]
        resolution: backbone input size (default 672; SAM3 native=1008).
                    Lower = faster but coarser features.
        returns: [f0, f1, f2] at resolution/3.5, resolution/7, resolution/14
        """
        if resolution is None:
            resolution = getattr(self, 'resolution', 672)

        images_norm = (images - 0.5) / 0.5

        # Resize to backbone resolution
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)

        out = self.backbone.forward_image(images_norm)
        fpn = out['backbone_fpn']
        return [f for f in fpn]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        images: (B, 3, H, W), range [0, 1]
        returns: (B, num_classes, H_out, W_out) logits
        """
        feats = self.extract_features(images)
        return self.decoder(feats)


def build_adapter_sam3_unet(sam3_model, adapter_bottleneck: int = 32,
                             num_classes: int = 5) -> AdapterSAM3UNet:
    """Factory function: build the complete Adapter-SAM3-UNet model."""
    model = AdapterSAM3UNet(sam3_model, adapter_bottleneck=adapter_bottleneck,
                            num_classes=num_classes)
    return model.cuda()


# ═══════════════════════════════════════════════════════════════
# DSM Dual-Stream Extension (Plan3 Route A+DSM)
# ═══════════════════════════════════════════════════════════════

class DSMEncoder(nn.Module):
    """Lightweight CNN encoder for single-channel DSM → 3-scale features.

    Matches SAM3 FPN scales (288, 144, 72) for fusion.
    ~0.05M params — negligible overhead.
    """

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, dsm: torch.Tensor) -> list:
        """dsm: (B, 1, H, W) or (B, H, W) → [f0, f1, f2] at 288², 144², 72²"""
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        x = self.enc(dsm)
        return [F.interpolate(x, (s, s), mode='bilinear', align_corners=False)
                for s in [288, 144, 72]]


class AdapterSAM3UNetDSM(AdapterSAM3UNet):
    """Dual-stream extension: SAM3 ViT (RGB) + CNN encoder (DSM) → fused UNet decoder.

    Adds ~0.1M trainable params on top of Route A's ~6.6M.
    """

    def __init__(self, sam3_model, adapter_bottleneck: int = 32, num_classes: int = 5,
                 dropout: float = 0.1, dsm_channels: int = 64):
        super().__init__(sam3_model, adapter_bottleneck=adapter_bottleneck,
                         num_classes=num_classes, dropout=dropout)

        self.dsm_encoder = DSMEncoder(out_ch=dsm_channels)

        # Fusion: concat RGB(256) + DSM(64) → 256 at each scale
        self.fuse2 = nn.Conv2d(256 + dsm_channels, 256, 1)
        self.fuse1 = nn.Conv2d(256 + dsm_channels, 256, 1)
        self.fuse0 = nn.Conv2d(256 + dsm_channels, 256, 1)

        dsm_trainable = sum(p.numel() for p in self.dsm_encoder.parameters())
        fusion_trainable = sum(p.numel() for p in
            [*self.fuse0.parameters(), *self.fuse1.parameters(), *self.fuse2.parameters()])
        print(f"  DSM encoder params: {dsm_trainable:,}")
        print(f"  Fusion params: {fusion_trainable:,}")

    def forward(self, images: torch.Tensor, dsm: torch.Tensor = None) -> torch.Tensor:
        """Forward with optional DSM. If dsm is None, falls back to RGB-only."""
        rgb_feats = self.extract_features(images)

        if dsm is not None:
            dsm_feats = self.dsm_encoder(dsm)
            # Fuse at each scale
            f0 = self.fuse0(torch.cat([rgb_feats[0], dsm_feats[0]], dim=1))
            f1 = self.fuse1(torch.cat([rgb_feats[1], dsm_feats[1]], dim=1))
            f2 = self.fuse2(torch.cat([rgb_feats[2], dsm_feats[2]], dim=1))
            feats = [f0, f1, f2]
        else:
            feats = rgb_feats

        return self.decoder(feats)
