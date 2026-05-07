#!/usr/bin/env python3
"""
Phase 3/4 升级版: SAM 3 特征提取 + UNet decoder 微调

相比 Phase 3 的简单 FPN decoder，使用更强的 UNet decoder:
- 更深的 decoder (5层 vs 4层)
- 更大的通道数 (256 vs 128)
- 更多 epochs (20 vs 5)
- 更好的数据增强
"""

import os, sys, json, time, gc, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import random

# Paths
sys.path.insert(0, '/root/Mynet/sam3-main')
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
for p in list(sys.path):
    if p == os.path.join(SE, 'sam3'):
        sys.path.remove(p)

from sam3.model_builder import build_sam3_image_model


# ============================================================
# UNet Decoder (stronger than Phase 3 FPN)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class SAM3UNetDecoder(nn.Module):
    """UNet-style decoder for SAM3 backbone features (3 scales: 288, 144, 72)."""
    def __init__(self, num_classes=5):
        super().__init__()
        # Encoder features: [0]=288²@256, [1]=144²@256, [2]=72²@256
        # Decoder: upsample + concat + conv blocks
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # 72→144
        self.conv1 = ConvBlock(512, 256)  # skip(256) + up(256) = 512
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # 144→288
        self.conv0 = ConvBlock(512, 256)  # skip(256) + up(256) = 512
        self.final_up = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 288→576
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, feats):
        """feats: [f0(288), f1(144), f2(72)] all (B,256,H,W)"""
        f0, f1, f2 = [f.float() for f in feats]
        x = self.up1(f2)  # 72→144
        x = self.conv1(torch.cat([x, f1], dim=1))  # 144: concat up+f1
        x = self.up2(x)  # 144→288
        x = self.conv0(torch.cat([x, f0], dim=1))  # 288: concat up+f0
        return self.final_up(x)  # 288→576→class_logits


# ============================================================
# Feature Extractor (SAM3 backbone, always frozen)
# ============================================================

class SAM3FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SAM3 backbone...")
        self.model = build_sam3_image_model(
            bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path=f'{SE}/weights/sam3/sam3.pt',
            device=device, eval_mode=True, enable_segmentation=True,
        )
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, images):
        """images: (B,3,H,W) → [(B,256,288,288), (B,256,144,144), (B,256,72,72)]"""
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != 1008 or w != 1008:
            images_norm = F.interpolate(images_norm, (1008, 1008), mode='bilinear', align_corners=False)
        out = self.model.backbone.forward_image(images_norm)
        fpn = out['backbone_fpn']
        feats = [fpn[k] for k in sorted(fpn.keys())] if isinstance(fpn, dict) else list(fpn[:3])
        # Convert to fp32, ensure on same device
        return [f.detach().float() for f in feats]

    def cleanup(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# Dataset
# ============================================================

class ISPRSTrainDataset(torch.utils.data.Dataset):
    CLASSES = ['road', 'building', 'grass', 'tree', 'car']
    LABEL_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:255}

    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix, crop=512, train=True):
        self.samples = []
        for tile in tiles:
            ip = os.path.join(img_dir, f'{tile}{img_suffix}')
            gp = os.path.join(gt_dir, f'{tile}{gt_suffix}')
            if os.path.exists(ip) and os.path.exists(gp):
                self.samples.append((ip, gp))
        self.crop = crop
        self.train = train
        self.n_crops = 80 if train else 10
        print(f"  Loaded {len(self.samples)} tiles")

    def __len__(self):
        return len(self.samples) * self.n_crops

    def __getitem__(self, idx):
        tile_idx = idx % len(self.samples)
        ip, gp = self.samples[tile_idx]
        img = np.array(Image.open(ip).convert('RGB'))
        label = np.array(Image.open(gp))

        # Remap labels
        rm = np.full_like(label, 255, dtype=np.int64)
        for k, v in self.LABEL_MAP.items():
            rm[label == k] = v

        if self.train:
            h, w = img.shape[:2]
            if h > self.crop and w > self.crop:
                y = random.randint(0, h - self.crop)
                x = random.randint(0, w - self.crop)
                img = img[y:y+self.crop, x:x+self.crop]
                rm = rm[y:y+self.crop, x:x+self.crop]
            # Augment
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); rm = np.fliplr(rm).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy(); rm = np.flipud(rm).copy()
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy(); rm = np.rot90(rm, k).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, torch.from_numpy(rm).long()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(decoder, extractor, val_loader, device):
    decoder.eval()
    intersect = torch.zeros(5, device=device)
    union = torch.zeros(5, device=device)
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        feats = extractor.extract(images)
        logits = decoder(feats)
        logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)

        mask = (labels != 255)
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(5):
            p_c = (pred == c); l_c = (labels == c)
            intersect[c] += (p_c & l_c).sum()
            union[c] += (p_c | l_c).sum()

    aAcc = correct / max(total, 1) * 100
    per_cls = [(intersect[c] / max(union[c], 1) * 100).item() for c in range(5)]
    miou = np.mean(per_cls)
    return aAcc, miou, per_cls


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='combined', choices=['vaihingen','potsdam','combined'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'unet_{args.dataset}'
    output_dir = os.path.join(args.output_dir, f'sam3_{tag}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Feature extractor
    extractor = SAM3FeatureExtractor(device)

    # Decoder
    decoder = SAM3UNetDecoder(num_classes=5).to(device)
    t = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {t:,}")

    # Data
    data_configs = {
        'vaihingen': {
            'img_dir': '/root/autodl-tmp/dataset/Vaihingen/top',
            'gt_dir': '/root/autodl-tmp/dataset/Vaihingen/gts_index',
            'img_suf': '.tif', 'gt_suf': '.png',
            'train_tiles': [f'top_mosaic_09cm_area{i}' for i in [1,3,5,7,11,13,15,17,21,23,26,28]],
            'val_tiles': [f'top_mosaic_09cm_area{i}' for i in [30,32,34,37]],
        },
        'potsdam': {
            'img_dir': '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB',
            'gt_dir': '/root/autodl-tmp/dataset/Potsdam/labels_index',
            'img_suf': '_RGB.tif', 'gt_suf': '.png',
            'train_tiles': [f'top_potsdam_{t}_{n}' for t in ['2','3','4','5'] for n in ['10','11','12']] +
                          [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['10','11']],
            'val_tiles': [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['7','8','9','12']],
        },
    }

    if args.dataset == 'combined':
        trains = []
        for ds in ['vaihingen', 'potsdam']:
            c = data_configs[ds]
            trains.append(ISPRSTrainDataset(c['img_dir'], c['gt_dir'], c['train_tiles'],
                                             c['img_suf'], c['gt_suf'], args.crop, True))
        train_ds = torch.utils.data.ConcatDataset(trains)
    else:
        c = data_configs[args.dataset]
        train_ds = ISPRSTrainDataset(c['img_dir'], c['gt_dir'], c['train_tiles'],
                                       c['img_suf'], c['gt_suf'], args.crop, True)

    # Val: Vaihingen
    cv = data_configs['vaihingen']
    v_val_ds = ISPRSTrainDataset(cv['img_dir'], cv['gt_dir'], cv['val_tiles'],
                                   cv['img_suf'], cv['gt_suf'], args.crop, False)
    # Val: Potsdam
    cp = data_configs['potsdam']
    p_val_ds = ISPRSTrainDataset(cp['img_dir'], cp['gt_dir'], cp['val_tiles'],
                                   cp['img_suf'], cp['gt_suf'], args.crop, False)

    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    v_loader = torch.utils.data.DataLoader(v_val_ds, 1, shuffle=False, num_workers=2, pin_memory=True)
    p_loader = torch.utils.data.DataLoader(p_val_ds, 1, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_v, best_p = 0.0, 0.0
    history = {'loss': [], 'v_miou': [], 'p_miou': []}

    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        t0 = time.time()

        for bi, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            feats = extractor.extract(images)
            logits = decoder(feats)
            logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if bi % 20 == 0:
                print(f"  E{epoch}/{args.epochs} B{bi}/{len(train_loader)} loss={loss.item():.4f}", end='\r')

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validate
        v_a, v_m, v_pc = validate(decoder, extractor, v_loader, device)
        p_a, p_m, p_pc = validate(decoder, extractor, p_loader, device)

        history['loss'].append(avg_loss)
        history['v_miou'].append(v_m)
        history['p_miou'].append(p_m)

        print(f"  E{epoch:3d}: loss={avg_loss:.4f} V={v_m:.1f}% P={p_m:.1f}% ({time.time()-t0:.0f}s)")

        if v_m > best_v:
            best_v = v_m
            torch.save({'epoch': epoch, 'decoder': decoder.state_dict(), 'v_miou': v_m, 'p_miou': p_m,
                        'v_per_class': v_pc}, os.path.join(output_dir, 'best_model.pt'))

    json.dump(history, open(os.path.join(output_dir, 'history.json'), 'w'))
    torch.save({'epoch': args.epochs, 'decoder': decoder.state_dict(), 'history': history},
               os.path.join(output_dir, 'final_model.pt'))
    extractor.cleanup()

    print(f"\nDone! Best V={best_v:.1f}% P={best_p:.1f}%")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
