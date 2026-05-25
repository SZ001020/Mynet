"""SAM-HQ style Global-Local Feature Fusion for Plan7-C3.

Fuses three feature sources via element-wise sum (SAM-HQ Table 3: sum > FPN):
  1. ViT block 6 (early, local boundary details)
  2. ViT block 32 / backbone_fpn[-1] (late, global context)
  3. MFNetDecoder b3 output (mask shape, decoder intermediate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GlobalLocalFusion(nn.Module):
    """Fuse early ViT + late ViT + decoder intermediate features.

    Mirrors SAM-HQ's compress_vit_feat + embedding_encoder + embedding_maskfeature.
    All three paths are upsampled to the same spatial resolution and summed.
    """

    def __init__(self, early_dim=1024, late_dim=256, decoder_dim=64, out_dim=64):
        super().__init__()
        # Early ViT: block 6 output, B×1024×64×64 → B×64×256×256
        self.compress_early = nn.Sequential(
            nn.ConvTranspose2d(early_dim, 256, kernel_size=2, stride=2),
            LayerNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, out_dim, kernel_size=2, stride=2),
        )
        # Late ViT: backbone_fpn[-1], B×256×64×64 → B×64×256×256
        self.compress_late = nn.Sequential(
            nn.ConvTranspose2d(late_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, out_dim, kernel_size=2, stride=2),
        )
        # Decoder: b3 output, B×64×H/16×W/16 → B×64×256×256
        self.compress_decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, 128, 3, 1, 1),
            LayerNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_dim, 3, 1, 1),
        )

    def forward(self, early_feat, late_feat, decoder_mid):
        """
        Args:
            early_feat:  [B, 1024, h, w]    from ViT block 6
            late_feat:   [B, 256, h, w]     from backbone_fpn[-1]
            decoder_mid: [B, 64, h/4, w/4]  from MFNetDecoder b3 output
        Returns:
            hq_features: [B, out_dim, 256, 256]
        """
        target_hw = (256, 256)

        f_early = self.compress_early(early_feat)
        f_late = self.compress_late(late_feat)
        f_decoder = self.compress_decoder(decoder_mid)

        # Align all to 256×256 (input resolutions vary with actual patch grid)
        if f_early.shape[-2:] != target_hw:
            f_early = F.interpolate(f_early, target_hw, mode="bilinear", align_corners=False)
        if f_late.shape[-2:] != target_hw:
            f_late = F.interpolate(f_late, target_hw, mode="bilinear", align_corners=False)
        if f_decoder.shape[-2:] != target_hw:
            f_decoder = F.interpolate(f_decoder, target_hw, mode="bilinear", align_corners=False)

        return f_early + f_late + f_decoder
