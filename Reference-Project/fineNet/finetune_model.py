"""
Phase 3: SAM 3 遥感微调模型

策略: 冻结 SAM 3 backbone，提取多尺度特征 → 训练轻量 FPN decoder。
用三种学习率/参数分组实现 partial/full/lora 的对比验证。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, '/root/Mynet/SegEarth-OV-3-main')
sys.path.insert(0, '/root/Mynet/sam3-main')

# Monkey-patch fused MLP before importing SAM 3
import sam3.perflib.fused as _fused
_orig_addmm_act = _fused.addmm_act
def _patched_addmm_act(act_type, weight, x):
    if torch.is_grad_enabled():
        out = F.linear(x, weight)
        return F.gelu(out) if act_type == 'gelu' else F.relu(out)
    return _orig_addmm_act(act_type, weight, x)
_fused.addmm_act = _patched_addmm_act

from sam3.model_builder import build_sam3_image_model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FPNHead(nn.Module):
    """Lightweight FPN decoder for 5-class segmentation."""
    def __init__(self, in_channels=256, num_classes=5):
        super().__init__()
        mid = 128
        self.lat3 = nn.Conv2d(in_channels, mid, 1)
        self.lat2 = nn.Conv2d(in_channels, mid, 1)
        self.lat1 = nn.Conv2d(in_channels, mid, 1)
        self.lat0 = nn.Conv2d(in_channels, mid, 1)
        self.sm3 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.sm2 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.sm1 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.sm0 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(mid, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1))

    def forward(self, feats):
        """feats: [f0(hi), f1, f2, f3(lo)] all (B,256,H,W)"""
        f0, f1, f2, f3 = feats
        p3 = self.sm3(self.lat3(f3))
        p2 = self.sm2(self.lat2(f2) + F.interpolate(p3, f2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.sm1(self.lat1(f1) + F.interpolate(p2, f1.shape[-2:], mode='bilinear', align_corners=False))
        p0 = self.sm0(self.lat0(f0) + F.interpolate(p1, f0.shape[-2:], mode='bilinear', align_corners=False))
        return self.head(p0)


class FineTunedSAM3(nn.Module):
    """
    SAM 3 backbone (frozen) + trainable FPN decoder.
    三种模式通过参数分组实现:
    - partial: decoder 正常训练
    - full: decoder + backbone 全部训练
    - lora: decoder 训练 + backbone LoRA
    """
    def __init__(self, num_classes=5, mode='partial', lora_rank=8, device='cuda'):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        self.sam3 = build_sam3_image_model(
            bpe_path='/root/Mynet/SegEarth-OV-3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path='/root/Mynet/SegEarth-OV-3-main/weights/sam3/sam3.pt',
            device=device,
        )
        self.decoder = FPNHead(in_channels=256, num_classes=num_classes).to(device)

        if mode == 'partial':
            for p in self.sam3.parameters():
                p.requires_grad = False
        elif mode == 'lora':
            for p in self.sam3.parameters():
                p.requires_grad = False
            self._inject_lora(rank=lora_rank)
        # full: all requires_grad (backbone + decoder)

        self.to(device)

    def _inject_lora(self, rank=8, alpha=16):
        """在 backbone attention 层中注入 LoRA."""
        scale = alpha / rank
        for name, module in self.sam3.named_modules():
            if isinstance(module, nn.Linear) and any(k in name.lower() for k in ('qkv', 'proj', 'fc')):
                module.lora_A = nn.Parameter(torch.zeros(module.in_features, rank).to(module.weight.device))
                module.lora_B = nn.Parameter(torch.zeros(rank, module.out_features).to(module.weight.device))
                nn.init.kaiming_uniform_(module.lora_A, a=5**0.5)
                nn.init.zeros_(module.lora_B)
                module.lora_scale = scale
                module._has_lora = True

                orig_forward = module.forward
                def make_lora_forward(lin, orig):
                    def lora_forward(x):
                        return orig(x) + (x @ lin.lora_A @ lin.lora_B) * lin.lora_scale
                    return lora_forward
                module.forward = make_lora_forward(module, orig_forward)

    def extract_features(self, images):
        """提取 SAM 3 backbone 多尺度特征 (必须 1008×1008 输入)."""
        images_norm = (images - 0.5) / 0.5
        # SAM 3 的 RoPE 预计算为 1008 分辨率
        _, _, h, w = images_norm.shape
        if h != 1008 or w != 1008:
            images_norm = F.interpolate(images_norm, size=(1008, 1008),
                                         mode='bilinear', align_corners=False)
        need_grad = self.mode in ('full', 'lora')
        ctx = torch.no_grad() if not need_grad else torch.enable_grad()
        with ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                backbone_out = self.sam3.backbone.forward_image(images_norm)

        # backbone_fpn has 3 levels: list or dict {0:(B,256,288,288), 1:(144,144), 2:(72,72)}
        fpn = backbone_out.get('backbone_fpn', [])
        if isinstance(fpn, dict):
            feats = [fpn[k].float() for k in sorted(fpn.keys())]
        elif isinstance(fpn, list) and len(fpn) >= 3:
            feats = [f.float() for f in fpn[:3]]
        else:
            raise ValueError(f"Cannot extract features from fpn type={type(fpn)}")
        # Add 4th level by downsampling
        f4 = F.max_pool2d(feats[-1], kernel_size=2, stride=2)
        feats.append(f4)
        return feats  # [f0(288), f1(144), f2(72), f3(36)]

    def forward(self, images):
        feats = self.extract_features(images)
        logits = self.decoder(feats)
        return F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
