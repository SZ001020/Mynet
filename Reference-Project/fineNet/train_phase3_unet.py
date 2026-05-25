#!/usr/bin/env python3
"""
Plan1 Phase 3: UNet Decoder Fine-tuning

纯 SegEarth SAM3 路径 — 冻结 backbone 提取特征 → 训练 UNet decoder。
解决之前 Phase 3 的 dtype 冲突和路径混乱问题。

架构: SAM3 ViTDet (frozen, no_grad) → backbone_fpn [3 scales]
      → UNet decoder (3-level) → 5-class logits
"""

import os, sys, json, time, gc, random, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# === ONLY SegEarth path (no os.chdir to avoid CUDA context issues) ===
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
# Import sam3 from SegEarth (uses relative paths internally, need cwd temporarily)
_prev_cwd = os.getcwd()
os.chdir(SE)
from sam3 import build_sam3_image_model
os.chdir(_prev_cwd)  # Restore immediately


# ============================================================
# UNet Decoder
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNetDecoder(nn.Module):
    """3-level UNet: 216→144→72 with skip connections."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # 72→144
        self.conv1 = ConvBlock(512, 256)
        self.up0 = nn.ConvTranspose2d(256, 256, 2, stride=2)  # 144→216 (use pad or interp)
        self.conv0 = ConvBlock(512, 256)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, feats):
        f0, f1, f2 = [f.float() for f in feats]  # (288,144,72)
        x = self.up1(f2)  # 72→144
        # Handle size mismatch
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, f1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, f1], dim=1))
        x = self.up0(x)  # 144→288
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, f0.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv0(torch.cat([x, f0], dim=1))
        return self.head(x)


# ============================================================
# SAM3 Extractor (SegEarth, frozen)
# ============================================================
class SAM3Extractor:
    def __init__(self, device='cuda'):
        print("Loading SAM3 (SegEarth)...")
        _cwd = os.getcwd(); os.chdir(SE)
        self.model = build_sam3_image_model(
            bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
        os.chdir(_cwd)  # Restore
        # Force CUDA: activation checkpointing may leave params on CPU
        self.model = self.model.cuda()
        for p in self.model.parameters():
            p.data = p.data.cuda()
            p.requires_grad = False
        # Verify
        devices = set(p.device.type for p in self.model.parameters())
        if 'cpu' in devices:
            # Last resort: reload with explicit CUDA
            self.model = self.model.to('cuda')
        print(f"  Model devices: {devices}")

    @torch.no_grad()
    def extract(self, images):
        """images: (B,3,H,W) → [(B,256,288,288), (B,256,144,144), (B,256,72,72)]"""
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != 1008 or w != 1008:
            images_norm = F.interpolate(images_norm, (1008, 1008), mode='bilinear', align_corners=False)
        out = self.model.backbone.forward_image(images_norm)
        fpn = out['backbone_fpn']
        feats = [fpn[k].detach() for k in sorted(fpn.keys())] if isinstance(fpn, dict) else [f.detach() for f in fpn[:3]]
        return feats

    def cleanup(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Dataset
# ============================================================
class ISPRSTrainDataset(torch.utils.data.Dataset):
    LABEL_MAP = {1:0,2:1,3:2,4:3,5:4,6:255}
    def __init__(self, img_dir, gt_dir, tiles, img_suf, gt_suf, crop=512, train=True):
        self.samples = [(os.path.join(img_dir,f'{t}{img_suf}'), os.path.join(gt_dir,f'{t}{gt_suf}'))
                        for t in tiles if os.path.exists(os.path.join(img_dir,f'{t}{img_suf}'))]
        self.crop, self.train, self.n = crop, train, 80 if train else 10
        print(f"  {len(self.samples)} tiles")
    def __len__(self): return len(self.samples) * self.n
    def __getitem__(self, idx):
        ip, gp = self.samples[idx % len(self.samples)]
        img = np.array(Image.open(ip).convert('RGB'))
        label = np.array(Image.open(gp))
        rm = np.full_like(label, 255, dtype=np.int64)
        for k,v in self.LABEL_MAP.items(): rm[label==k]=v
        if self.train:
            h,w=img.shape[:2]
            if h>self.crop and w>self.crop:
                y=random.randint(0,h-self.crop); x=random.randint(0,w-self.crop)
                img=img[y:y+self.crop,x:x+self.crop]; rm=rm[y:y+self.crop,x:x+self.crop]
            if random.random()<0.5: img=np.fliplr(img).copy(); rm=np.fliplr(rm).copy()
            if random.random()<0.5: img=np.flipud(img).copy(); rm=np.flipud(rm).copy()
            k=random.randint(0,3)
            if k>0: img=np.rot90(img,k).copy(); rm=np.rot90(rm,k).copy()
        return torch.from_numpy(img).permute(2,0,1).float()/255.0, torch.from_numpy(rm).long()


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(decoder, extractor, loader, device):
    decoder.eval()
    inter=torch.zeros(5,device=device); union=torch.zeros(5,device=device); corr=0; tot=0
    for img, lab in loader:
        img,lab=img.to(device),lab.to(device)
        f=extractor.extract(img)
        logits=decoder(f)
        logits=F.interpolate(logits,lab.shape[-2:],mode='bilinear',align_corners=False)
        pred=logits.argmax(1)
        mask=(lab!=255); corr+=(pred[mask]==lab[mask]).sum().item(); tot+=mask.sum().item()
        for c in range(5):
            pc=(pred==c); lc=(lab==c); inter[c]+=(pc&lc).sum(); union[c]+=(pc|lc).sum()
    aAcc=corr/max(tot,1)*100
    pc=[(inter[c]/max(union[c],1)*100).item() for c in range(5)]
    return aAcc, np.mean(pc), pc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'phase3_unet_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    # Model
    extractor = SAM3Extractor(device)
    decoder = UNetDecoder(5).to(device)
    t = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder: {t:,} params")

    # Data
    VAIH = '/root/autodl-tmp/dataset/Vaihingen'
    POTS = '/root/autodl-tmp/dataset/Potsdam'
    v_train = [f'top_mosaic_09cm_area{i}' for i in [1,3,5,7,11,13,15,17,21,23,26,28]]
    v_val   = [f'top_mosaic_09cm_area{i}' for i in [30,32,34,37]]
    p_train = [f'top_potsdam_{t}_{n}' for t in ['2','3','4','5'] for n in ['10','11','12']] + \
              [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['10','11']]
    p_val   = [f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['7','8','9','12']]

    train_ds = torch.utils.data.ConcatDataset([
        ISPRSTrainDataset(VAIH+'/top', VAIH+'/gts_index', v_train, '.tif','.png', args.crop, True),
        ISPRSTrainDataset(POTS+'/2_Ortho_RGB', POTS+'/labels_index', p_train, '_RGB.tif','.png', args.crop, True),
    ])
    v_ldr = torch.utils.data.DataLoader(
        ISPRSTrainDataset(VAIH+'/top', VAIH+'/gts_index', v_val, '.tif','.png', args.crop, False), 1, num_workers=0)
    p_ldr = torch.utils.data.DataLoader(
        ISPRSTrainDataset(POTS+'/2_Ortho_RGB', POTS+'/labels_index', p_val, '_RGB.tif','.png', args.crop, False), 1, num_workers=0)
    train_ldr = torch.utils.data.DataLoader(train_ds, args.batch, shuffle=True, num_workers=0, drop_last=True)

    # Optimizer
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    crit = nn.CrossEntropyLoss(ignore_index=255)

    best_v, best_p = 0.0, 0.0
    hist = {'loss':[], 'v_miou':[], 'p_miou':[]}

    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(1, args.epochs+1):
        decoder.train()
        ep_loss, nb = 0.0, 0; t0 = time.time()
        for bi, (img, lab) in enumerate(train_ldr):
            img, lab = img.to(device), lab.to(device)
            feats = extractor.extract(img)
            logits = decoder(feats)
            logits = F.interpolate(logits, lab.shape[-2:], mode='bilinear', align_corners=False)
            loss = crit(logits, lab)
            opt.zero_grad(); scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item(); nb += 1
            if bi % 20 == 0: print(f"  E{epoch}/{args.epochs} B{bi}/{len(train_ldr)} loss={loss.item():.4f}", end='\r')
        sch.step()
        avg_loss = ep_loss/max(nb,1)
        v_a, v_m, v_pc = validate(decoder, extractor, v_ldr, device)
        p_a, p_m, p_pc = validate(decoder, extractor, p_ldr, device)
        hist['loss'].append(avg_loss); hist['v_miou'].append(v_m); hist['p_miou'].append(p_m)
        print(f"  E{epoch:3d}: loss={avg_loss:.4f} V={v_m:.1f}% P={p_m:.1f}%  ({time.time()-t0:.0f}s)")
        if v_m > best_v:
            best_v = v_m
            torch.save({'epoch':epoch,'decoder':decoder.state_dict(),'v_miou':v_m,'p_miou':p_m,'v_per_class':v_pc,'p_per_class':p_pc},
                       os.path.join(out_dir,'best_model.pt'))

    extractor.cleanup()
    json.dump(hist, open(os.path.join(out_dir,'history.json'),'w'))
    print(f"\nDone! Best V={best_v:.1f}% P={best_p:.1f}%\nOutput: {out_dir}")

if __name__ == '__main__': main()
