"""A3 variant with softmax gate (B1 ablation). Patches sigmoid→softmax."""
from __future__ import annotations
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
BASE = "/root/Mynet"
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation")
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter")
from simple_decoder import SimpleDecoder
from mm_adapter_vit import inject_mm_adapters, DSMTokenEncoder

class Pyramid4Scale(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ch,ch,2,2),nn.BatchNorm2d(ch),nn.ReLU(inplace=False))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(ch,ch,2,2),nn.BatchNorm2d(ch),nn.ReLU(inplace=False),
                                  nn.ConvTranspose2d(ch,ch,2,2),nn.BatchNorm2d(ch),nn.ReLU(inplace=False))
        self.id3 = nn.Identity(); self.down = nn.MaxPool2d(2,2)
    def forward(self, feat):
        return [self.up4(feat), self.up2(feat), self.id3(feat), self.down(feat)]


class A3SoftmaxGate(nn.Module):
    """A3 + softmax gate (B1). Patches all MMAdapter blocks to use softmax after injection."""

    def __init__(self, sam3_model, num_classes=5, bottleneck=32, dsm_dim=128):
        super().__init__()
        self.backbone = sam3_model.backbone; self.resolution = 1008
        self.mm_state = inject_mm_adapters(self.backbone.vision_backbone, bottleneck=bottleneck,
                                           dsm_attn_mode="full", checkpoint_attn=True)
        self.dsm_encoder = DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.pyramid = Pyramid4Scale(256)
        self.decoder = SimpleDecoder(in_channels=256, decode_channels=64, num_classes=num_classes)
        if hasattr(self.backbone, "language_backbone"):
            for param in self.backbone.language_backbone.parameters(): param.requires_grad = False
        self._patch_softmax_gates()
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  B1 softmax gate: {t:,} trainable / {total:,} total")

    def _patch_softmax_gates(self):
        """Replace sigmoid wx/wy gates with softmax [w_rgb, w_dsm] in each block."""
        trunk = self.backbone.vision_backbone.trunk
        patched = 0
        for blk in trunk.blocks:
            inner = blk.block if hasattr(blk, 'block') else blk
            if hasattr(inner, 'wx_logit'):
                inner._use_softmax = True
                inner.softmax_logits = nn.Parameter(torch.zeros(2))
                inner.softmax_logits.requires_grad = True
                # Freeze old sigmoid gates
                inner.wx_logit.requires_grad = False
                inner.wy_logit.requires_grad = False
                patched += 1
        print(f"  Patched {patched} blocks: sigmoid → softmax gate")

    # Override forward to use patched gate
    def _patched_forward(self, x):
        """Same as MMAdapterBlock.forward but with softmax gate."""
        # We need to access the blocks' forward methods. Since we patched in-place,
        # we monkey-patch each block's forward to use softmax.
        pass

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution,self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution,self.resolution), mode="bilinear", align_corners=False)
        patch_hw = (self.resolution // 16, self.resolution // 16)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        try:
            out = self.backbone.forward_image(images)
        finally:
            self.mm_state.pop("dsm_tokens", None)
        feat = out["backbone_fpn"][-1].clone()
        scales = self.pyramid(feat)
        logits = self.decoder(scales[-1])
        return F.interpolate(logits, (256,256), mode="bilinear", align_corners=False)
