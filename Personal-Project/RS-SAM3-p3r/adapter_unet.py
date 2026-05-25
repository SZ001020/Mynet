"""
Plan3 Route A+DSM (p3r): Deep DSM Fusion + UNetFormer Decoder

Upgrades over RS-SAM3-p3:
1. Deep DSM fusion: DSM injected at every ViT block (AdapterBlockDSM)
   instead of post-hoc concat after FPN
2. UNetFormer decoder: GLA window attention replacing simple ConvBlocks

Model variants:
- AdapterSAM3UNetFormer: RGB-only (Route A with better decoder)
- AdapterSAM3UNetFormerDSM: RGB+DSM deep fusion (Route A+DSM with better decoder)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from adapter_vit import inject_adapters, DSMEncoderDeep
from unetformer_decoder import UNetFormerDecoder


class AdapterSAM3UNetFormer(nn.Module):
    """Route A with UNetFormer decoder (no DSM)."""

    def __init__(self, sam3_model, adapter_bottleneck=32, num_classes=5,
                 dropout=0.1, window_size=8, decode_channels=256):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.num_classes = num_classes

        # RGB-only adapters (no DSM)
        inject_adapters(self.backbone.vision_backbone, bottleneck=adapter_bottleneck,
                        use_dsm=False)

        for n, p in self.backbone.vision_backbone.named_parameters():
            if 'prompt_learn' not in n:
                p.requires_grad = False

        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        self.decoder = UNetFormerDecoder(num_classes=num_classes,
                                         decode_channels=decode_channels, dropout=dropout,
                                         window_size=window_size)
        self._log_params("RGB-only")

    def _log_params(self, tag):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        adapter = sum(p.numel() for n, p in self.backbone.vision_backbone.named_parameters()
                      if 'prompt_learn' in n)
        decoder = sum(p.numel() for p in self.decoder.parameters())
        print(f"  [{tag}] Adapter: {adapter:,} | Decoder: {decoder:,}")
        print(f"  [{tag}] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def extract_features(self, images, resolution=None):
        if resolution is None:
            resolution = getattr(self, 'resolution', 1008)
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)
        out = self.backbone.forward_image(images_norm)
        # Clone to decouple from SAM3's activation checkpointing memory reuse
        return [f.clone() for f in out['backbone_fpn']]

    def forward(self, images):
        feats = self.extract_features(images)
        return self.decoder(feats)


class AdapterSAM3UNetFormerDSM(AdapterSAM3UNetFormer):
    """Route A+DSM with deep DSM fusion + UNetFormer decoder.

    Overrides adapter injection to use AdapterBlockDSM (with DSM projections).
    """

    def __init__(self, sam3_model, adapter_bottleneck=32, num_classes=5,
                 dropout=0.1, window_size=8, dsm_dim=128, decode_channels=256):
        # Inject DSM-enabled adapters BEFORE parent init
        inject_adapters(sam3_model.backbone.vision_backbone,
                        bottleneck=adapter_bottleneck, use_dsm=True)

        # Skip parent's inject_adapters call by directly calling nn.Module.__init__
        # and manually setting up components
        nn.Module.__init__(self)
        self.backbone = sam3_model.backbone
        self.num_classes = num_classes

        # Freeze non-adapter params
        for n, p in self.backbone.vision_backbone.named_parameters():
            if 'prompt_learn' not in n and 'dsm_proj' not in n:
                p.requires_grad = False
        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        self.decoder = UNetFormerDecoder(num_classes=num_classes,
                                         decode_channels=decode_channels, dropout=dropout,
                                         window_size=window_size)
        self._log_params("RGB+DSM")

        # DSM encoder
        self.dsm_encoder = DSMEncoderDeep(dsm_dim=dsm_dim)
        # Project DSM scales (128ch) to match FPN (256ch) for residual add
        self.dsm_scale_proj = nn.ModuleList([
            nn.Conv2d(dsm_dim, 256, 1) for _ in range(3)
        ])
        dsm_params = sum(p.numel() for p in self.dsm_encoder.parameters())
        dsm_proj_params = sum(p.numel() for n, p in self.named_parameters()
                              if 'dsm_proj' in n)
        print(f"  DSM encoder: {dsm_params:,} | DSM projections: {dsm_proj_params:,}")

    def forward(self, images, dsm=None):
        if dsm is None:
            return super().forward(images)

        images_norm = (images - 0.5) / 0.5
        resolution = getattr(self, 'resolution', 1008)
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)

        # DSM encoding: 2D feature map for per-block spatial-adaptive fusion
        dsm_out = self.dsm_encoder(dsm)
        dsm_map = dsm_out['features']  # (B, dsm_dim, H/16, W/16) — 2D for spatial pooling

        # Broadcast DSM feature map to all AdapterBlockDSM
        for blk in self.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, '_dsm_map'):
                object.__setattr__(blk, '_dsm_map', dsm_map)

        out = self.backbone.forward_image(images_norm)
        fpn = out['backbone_fpn']

        # Clear after forward
        for blk in self.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, '_dsm_map'):
                object.__setattr__(blk, '_dsm_map', None)

        # DSM scale residual (already cloned in extract_features, safe to modify)
        dsm_scales = dsm_out['scales']
        feats = [f + self.dsm_scale_proj[i](dsm_scales[i]) for i, f in enumerate(fpn)]

        return self.decoder(feats)
