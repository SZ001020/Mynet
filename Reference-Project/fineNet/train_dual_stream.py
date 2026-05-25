#!/usr/bin/env python3
"""
Phase 4 深入: 双流多模态 SAM3 + DSM

RGB → SAM3 backbone (frozen, no_grad) → f_rgb [3 scales, 256ch]
DSM → Light CNN encoder (trainable)      → f_dsm [3 scales, 64ch]
         ↓ concat at each scale
    UNet decoder → 5-class logits

纯 SegEarth SAM3 路径, 无 sam3-main 依赖.
"""

import os, sys, json, time, gc, random, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_opening

# === Path: ONLY SegEarth ===
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
os.chdir(SE)
from sam3 import build_sam3_image_model


# ============================================================
# DSM Encoder
# ============================================================

class DSMEncoder(nn.Module):
    """Lightweight encoder: DSM (H,W) → multi-scale features matching SAM3 FPN."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # 3 scale heads — match SAM3 FPN: 288, 144, 72
        self.head0 = nn.Sequential(nn.Conv2d(64, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.head1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.head2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, dsm):
        """dsm: (B, H, W) → [(B,64,288,288), (B,64,144,144), (B,64,72,72)]"""
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        x = self.stem(dsm)  # (B,64,H/4,W/4)

        f0 = self.head0(x)
        if f0.shape[-2] != 288 or f0.shape[-1] != 288:
            f0 = F.interpolate(f0, (288, 288), mode='bilinear', align_corners=False)

        f1 = self.head1(self.down1(f0))
        if f1.shape[-2] != 144:
            f1 = F.interpolate(f1, (144, 144), mode='bilinear', align_corners=False)

        f2 = self.head2(self.down2(f1))
        if f2.shape[-2] != 72:
            f2 = F.interpolate(f2, (72, 72), mode='bilinear', align_corners=False)

        return [f0, f1, f2]


# ============================================================
# Dual-Stream UNet Decoder
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class DualStreamDecoder(nn.Module):
    """UNet decoder for RGB(256ch) + DSM(64ch) = 320ch features at 3 scales."""
    def __init__(self, num_classes=5, dsm_ch=64):
        super().__init__()
        rgb_ch = 256
        in_ch = rgb_ch + dsm_ch  # 320

        self.up1 = nn.ConvTranspose2d(in_ch, 256, 2, stride=2)  # 72→144
        self.conv1 = ConvBlock(256 + in_ch, 256)  # skip(320) + up(256) = 576

        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # 144→288
        self.conv0 = ConvBlock(256 + in_ch, 256)  # skip(320) + up(256) = 576

        self.final_up = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 288→576
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, rgb_feats, dsm_feats):
        """rgb_feats: [f0(288),f1(144),f2(72)] all 256ch
           dsm_feats: [d0(288),d1(144),d2(72)] all 64ch"""
        f = [torch.cat([r, d], dim=1) for r, d in zip(rgb_feats, dsm_feats)]
        f0, f1, f2 = f  # all (B, 320, Hi, Wi)

        x = self.up1(f2)  # 72→144, 256ch
        x = self.conv1(torch.cat([x, f1], dim=1))  # 144
        x = self.up2(x)  # 144→288, 256ch
        x = self.conv0(torch.cat([x, f0], dim=1))  # 288
        return self.final_up(x)  # 288→576→class_logits


# ============================================================
# SAM3 Feature Extractor (SegEarth, frozen)
# ============================================================

class SAM3Extractor:
    """使用 Sam3Processor (Phase 1 验证过的路径) 提取 backbone 特征."""
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SAM3 (via Sam3Processor)...")
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(
            bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
        # Force ALL params to CUDA (activation checkpointing may leave some on CPU)
        model = model.cuda()
        for p in model.parameters():
            p.data = p.data.cuda()
        self.processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

    @torch.no_grad()
    def extract(self, images):
        """images: (B,3,H,W) → [(B,256,288,288), (B,256,144,144), (B,256,72,72)]"""
        # Use set_image → backbone_out extraction (processor path handles devices correctly)
        feats_list = []
        for i in range(images.shape[0]):
            img = images[i]  # (3, H, W)
            # Convert to PIL via numpy
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            # Ensure model is on CUDA before each forward (activation ckpt may move params)
            self.processor.model = self.processor.model.cuda()
            state = self.processor.set_image(pil_img)
            backbone_out = state['backbone_out']
            fpn = backbone_out['backbone_fpn']
            f = [fpn[k].detach() for k in sorted(fpn.keys())] if isinstance(fpn, dict) else [t.detach() for t in fpn[:3]]
            feats_list.append(f)
        # Stack: list of 3 → each (B, C, H, W)
        return [torch.cat([fl[i] for fl in feats_list], dim=0).to(self.device) for i in range(3)]

    def cleanup(self):
        del self.processor; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Dataset
# ============================================================

class DualStreamDataset(torch.utils.data.Dataset):
    LABEL_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:255}

    def __init__(self, img_dir, gt_dir, dsm_dir, tiles, img_suf, gt_suf, dsm_pattern, crop=512, train=True):
        self.samples = []
        for tile in tiles:
            ip = os.path.join(img_dir, f'{tile}{img_suf}')
            gp = os.path.join(gt_dir, f'{tile}{gt_suf}')
            # Try to find matching DSM file
            dp = None
            # Vaihingen: top_mosaic_09cm_area1 → dsm_09cm_matching_area1
            # Potsdam: top_potsdam_2_10 → dsm_potsdam_2_10
            for variant in [tile, tile.replace('top_mosaic_09cm_', ''),
                            tile.replace('top_potsdam_', 'potsdam_')]:
                candidate = os.path.join(dsm_dir, dsm_pattern.format(tile=variant))
                if os.path.exists(candidate):
                    dp = candidate
                    break
            if os.path.exists(ip) and os.path.exists(gp):
                self.samples.append((ip, gp, dp))
        self.crop = crop
        self.train = train
        self.n_crops = 80 if train else 10
        print(f"  Loaded {len(self.samples)} tiles ({sum(1 for _,_,d in self.samples if d)} with DSM)")

    def __len__(self): return len(self.samples) * self.n_crops

    def __getitem__(self, idx):
        tile_idx = idx % len(self.samples)
        ip, gp, dp = self.samples[tile_idx]

        img = np.array(Image.open(ip).convert('RGB'))
        label = np.array(Image.open(gp))
        rm = np.full_like(label, 255, dtype=np.int64)
        for k, v in self.LABEL_MAP.items():
            rm[label == k] = v

        # Load DSM
        if dp:
            dsm = np.array(Image.open(dp)).astype(np.float32)
            # Compute nDSM
            try:
                from scipy.ndimage import grey_opening
                ground = grey_opening(dsm, size=101)
                ndsm = dsm - ground
            except:
                ndsm = dsm - dsm.min()
            dsm_norm = np.clip(ndsm / 10.0, 0, 1)
        else:
            dsm_norm = np.zeros(img.shape[:2], dtype=np.float32)

        if self.train:
            h, w = img.shape[:2]
            if h > self.crop and w > self.crop:
                y = random.randint(0, h - self.crop)
                x = random.randint(0, w - self.crop)
                img = img[y:y+self.crop, x:x+self.crop]
                rm = rm[y:y+self.crop, x:x+self.crop]
                dsm_norm = dsm_norm[y:y+self.crop, x:x+self.crop]
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); rm = np.fliplr(rm).copy(); dsm_norm = np.fliplr(dsm_norm).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy(); rm = np.flipud(rm).copy(); dsm_norm = np.flipud(dsm_norm).copy()
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy(); rm = np.rot90(rm, k).copy(); dsm_norm = np.rot90(dsm_norm, k).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        dsm_t = torch.from_numpy(dsm_norm).float()
        return img_t, dsm_t, torch.from_numpy(rm).long()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(decoder, dsm_encoder, extractor, val_loader, device):
    decoder.eval(); dsm_encoder.eval()
    intersect = torch.zeros(5, device=device)
    union = torch.zeros(5, device=device)
    correct = 0; total = 0

    for images, dsms, labels in val_loader:
        images, dsms, labels = images.to(device), dsms.to(device), labels.to(device)
        rgb_f = extractor.extract(images)
        dsm_f = dsm_encoder(dsms)
        logits = decoder(rgb_f, dsm_f)
        logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)
        mask = labels != 255
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(5):
            pc, lc = (pred == c), (labels == c)
            intersect[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()

    aAcc = correct / max(total, 1) * 100
    per_cls = [(intersect[c] / max(union[c], 1) * 100).item() for c in range(5)]
    return aAcc, np.mean(per_cls), per_cls


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'phase4_dual_stream_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # SAM3 extractor
    extractor = SAM3Extractor(device)

    # DSM encoder + decoder
    dsm_encoder = DSMEncoder(out_channels=64).to(device)
    decoder = DualStreamDecoder(num_classes=5, dsm_ch=64).to(device)
    t_dsm = sum(p.numel() for p in dsm_encoder.parameters())
    t_dec = sum(p.numel() for p in decoder.parameters())
    print(f"DSM encoder: {t_dsm:,} params | Decoder: {t_dec:,} params")

    # Data
    data = {
        'vaihingen': {
            'img_dir': '/root/autodl-tmp/dataset/Vaihingen/top',
            'gt_dir': '/root/autodl-tmp/dataset/Vaihingen/gts_index',
            'dsm_dir': '/root/autodl-tmp/dataset/Vaihingen/dsm',
            'dsm_pattern': 'dsm_09cm_matching_{tile}.tif',
            'img_suf': '.tif', 'gt_suf': '.png',
            'train': [f'top_mosaic_09cm_area{i}' for i in [1,3,5,7,11,13,15,17,21,23,26,28]],
            'val':   [f'top_mosaic_09cm_area{i}' for i in [30,32,34,37]],
        },
        'potsdam': {
            'img_dir': '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB',
            'gt_dir': '/root/autodl-tmp/dataset/Potsdam/labels_index',
            'dsm_dir': '/root/autodl-tmp/dataset/Potsdam/1_DSM',
            'dsm_pattern': 'dsm_{tile}.tif',
            'img_suf': '_RGB.tif', 'gt_suf': '.png',
            'train': [f'top_potsdam_{t}_{n}' for t in ['2','3','4','5'] for n in ['10','11','12']] +
                      [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['10','11']],
            'val':   [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['7','8','9','12']],
        },
    }

    # Fix DSM patterns for tile name differences
    data['vaihingen']['dsm_pattern'] = 'dsm_09cm_matching_{tile}.tif'
    data['potsdam']['dsm_pattern'] = 'dsm_{tile}.tif'
    # Fix tile naming for DSM
    for ds_name in ['vaihingen', 'potsdam']:
        d = data[ds_name]
        d['train_ds'] = DualStreamDataset(d['img_dir'], d['gt_dir'], d['dsm_dir'],
                                           d['train'], d['img_suf'], d['gt_suf'],
                                           d['dsm_pattern'], args.crop, True)
        d['val_ds'] = DualStreamDataset(d['img_dir'], d['gt_dir'], d['dsm_dir'],
                                         d['val'], d['img_suf'], d['gt_suf'],
                                         d['dsm_pattern'], args.crop, False)

    # Combined train
    train_ds = torch.utils.data.ConcatDataset([data['vaihingen']['train_ds'], data['potsdam']['train_ds']])
    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    v_loader = torch.utils.data.DataLoader(data['vaihingen']['val_ds'], 1, shuffle=False, num_workers=0)
    p_loader = torch.utils.data.DataLoader(data['potsdam']['val_ds'], 1, shuffle=False, num_workers=0)

    # Optimizer
    all_params = list(dsm_encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.amp.GradScaler('cuda')

    best_v, best_p = 0.0, 0.0
    history = {'loss': [], 'v_miou': [], 'p_miou': []}

    print(f"\n{'='*60}")
    print(f"Phase 4 Dual-Stream: RGB + DSM ({args.epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        decoder.train(); dsm_encoder.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for images, dsms, labels in train_loader:
            images, dsms, labels = images.to(device), dsms.to(device), labels.to(device)

            rgb_f = extractor.extract(images)
            dsm_f = dsm_encoder(dsms)
            logits = decoder(rgb_f, dsm_f)
            logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item(); n_batches += 1
            if n_batches % 20 == 0:
                print(f"  E{epoch}/{args.epochs} B{n_batches}/{len(train_loader)} loss={loss.item():.4f}", end='\r')

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        v_a, v_m, v_pc = validate(decoder, dsm_encoder, extractor, v_loader, device)
        p_a, p_m, p_pc = validate(decoder, dsm_encoder, extractor, p_loader, device)

        history['loss'].append(avg_loss)
        history['v_miou'].append(v_m)
        history['p_miou'].append(p_m)
        print(f"  E{epoch:3d}: loss={avg_loss:.4f} V={v_m:.1f}% P={p_m:.1f}%  ({time.time()-t0:.0f}s)")

        if v_m > best_v:
            best_v = v_m
            torch.save({'epoch': epoch, 'dsm_encoder': dsm_encoder.state_dict(),
                        'decoder': decoder.state_dict(), 'v_miou': v_m, 'p_miou': p_m,
                        'v_per_class': v_pc, 'p_per_class': p_pc},
                       os.path.join(output_dir, 'best_model.pt'))

        if epoch % 3 == 0:
            torch.save({'epoch': epoch, 'dsm_encoder': dsm_encoder.state_dict(),
                        'decoder': decoder.state_dict(), 'history': history},
                       os.path.join(output_dir, f'ckpt_epoch{epoch}.pt'))

    json.dump(history, open(os.path.join(output_dir, 'history.json'), 'w'))
    extractor.cleanup()

    print(f"\nDone! Best: V={best_v:.1f}% P={best_p:.1f}%")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
