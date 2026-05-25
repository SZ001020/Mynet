#!/usr/bin/env python3
"""Plan5 v2: SAM_RS weights (0.01) + ColorJitter + car×5 weighting."""
import os, sys, json, time, gc, argparse, random
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageEnhance

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p5'); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM-p3b')
from structure_loss import structure_loss
from adapter_vit import inject_adapters
from mfnet_decoder import MFNetDecoder, Pyramid4Scale, SEFusion
from dataset_adapter import (NUM_CLASSES, IGNORE_INDEX, VAIHINGEN_TRAIN, VAIHINGEN_VAL)

def boundary_loss(logits, bmap):
    pred = torch.sigmoid(logits.max(1, keepdim=True)[0])
    b = bmap.unsqueeze(1).float()
    inter = (pred * b).sum(); union = pred.sum() + b.sum() - inter
    return 1.0 - (inter + 1) / (union + 1)

def object_loss(logits, omap):
    pred = F.softmax(logits, 1); obj = (omap > 0).float()
    loss = 0.0
    for c in range(5):
        pc = pred[:, c]; mc = (omap > 0).float()
        inter = (pc * mc).sum(); union = pc.sum() + mc.sum() - inter
        loss += 1.0 - (inter + 1) / (union + 1)
    return loss / 5

class VPT_MFNetDecoder(nn.Module):
    def __init__(self, sam3_model, adapter_bottleneck=32, num_classes=5, use_dsm=True, dropout=0.1):
        super().__init__()
        self.backbone = sam3_model.backbone; self.use_dsm = use_dsm
        inject_adapters(self.backbone.vision_backbone, bottleneck=adapter_bottleneck)
        for n, p in self.backbone.vision_backbone.named_parameters(): p.requires_grad = 'prompt_learn' in n
        for p in self.backbone.language_backbone.parameters(): p.requires_grad = False
        self.pyramid_rgb = Pyramid4Scale(256)
        if use_dsm:
            from adapter_vit import DSMEncoderDeep
            self.dsm_encoder = DSMEncoderDeep(dsm_dim=128); self.pyramid_dsm = Pyramid4Scale(128)
            self.f1, self.f2, self.f3, self.f4 = SEFusion(256), SEFusion(256), SEFusion(256), SEFusion(256)
            self.dsm_proj = nn.ModuleList([nn.Conv2d(128, 256, 1) for _ in range(4)])
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64, dropout=dropout)
        print(f"  Trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, images, dsm=None):
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != 1008 or w != 1008: images_norm = F.interpolate(images_norm, (1008, 1008), mode='bilinear', align_corners=False)
        out = self.backbone.forward_image(images_norm); vit_feat = out['backbone_fpn'][-1].clone()
        scales_rgb = self.pyramid_rgb(vit_feat)
        if dsm is not None and self.use_dsm:
            dsm_out = self.dsm_encoder(dsm); dsm_feat = dsm_out['features']
            if dsm_feat.shape[-2:] != vit_feat.shape[-2:]: dsm_feat = F.interpolate(dsm_feat, vit_feat.shape[-2:], mode='bilinear', align_corners=False)
            scales_dsm = self.pyramid_dsm(dsm_feat)
            feats = [f(scales_rgb[i], self.dsm_proj[i](scales_dsm[i])) for i, f in enumerate([self.f1, self.f2, self.f3, self.f4])]
        else: feats = scales_rgb
        return self.decoder(feats)

from train_256 import Window256Dataset, Window256DatasetDSM

class Window256DatasetBO(Window256DatasetDSM):
    def __init__(self, img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_tiles, is_train=True, stride=128):
        super().__init__(img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_tiles, is_train, stride)
        bo_dir = '/root/autodl-tmp/dataset/Vaihingen/boundary_object'
        self.boundaries, self.objects = [], []
        for tile in tiles:
            npz = f'{bo_dir}/{tile}.npz'
            if os.path.exists(npz):
                d = np.load(npz); self.boundaries.append(d['boundary']); self.objects.append(d['objects'])
            else:
                self.boundaries.append(np.zeros((256, 256), dtype=np.uint8)); self.objects.append(np.zeros((256, 256), dtype=np.int32))
        print(f"  {len(self.boundaries)} boundary/object maps loaded")

    def __getitem__(self, idx):
        img, dsm, lbl = super().__getitem__(idx)
        ti = idx % len(self.boundaries); b = self.boundaries[ti]; o = self.objects[ti]
        if self.is_train:
            h, w = b.shape[:2]
            rng = random.Random(idx); y = rng.randint(0, max(1, h - 256)); x = rng.randint(0, max(1, w - 256))
            b = b[y:y+256, x:x+256] if h > 256 else b; o = o[y:y+256, x:x+256] if h > 256 else o
        # ColorJitter augmentation
        if self.is_train and random.random() < 0.3:
            img_pil = Image.fromarray((img.permute(1,2,0).numpy()*255).astype(np.uint8))
            enhancer = random.choice([
                lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
                lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
                lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
            ])
            img_pil = enhancer(img_pil)
            img = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float()/255.0

        return img, dsm, lbl, torch.from_numpy(b.astype(np.float32)), torch.from_numpy(o.astype(np.int32) % 100)

def load_sam3():
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                               checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev); return m.cuda()

@torch.no_grad()
def validate(model, loader, device):
    model.eval(); inter = torch.zeros(5, device=device); union = torch.zeros(5, device=device); correct, total = 0, 0
    for batch in loader:
        if len(batch) == 5: images, dsm, labels, _, _ = batch
        else: images, dsm, labels = batch
        images, labels, dsm = images.to(device), labels.to(device), dsm.to(device)
        logits = model(images, dsm); logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1); mask = (labels != IGNORE_INDEX)
        correct += (pred[mask] == labels[mask]).sum().item(); total += mask.sum().item()
        for c in range(5): pc = (pred == c); lc = (labels == c); inter[c] += (pc & lc).sum(); union[c] += (pc | lc).sum()
    return correct / max(total, 1) * 100, np.mean([(inter[c] / max(union[c], 1) * 100).item() for c in range(5)]), [(inter[c] / max(union[c], 1) * 100).item() for c in range(5)]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--output', default='/root/autodl-tmp/runs'); args = p.parse_args()
    device = torch.device('cuda'); ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan5_{ts}'); os.makedirs(out_dir, exist_ok=True)
    print(f"Plan5: +boundary/object loss\n  Epochs: {args.epochs}, Batch: {args.batch}\n  Output: {out_dir}")

    print("\nLoading data..."); train_t, val_t = VAIHINGEN_TRAIN, VAIHINGEN_VAL
    img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'; gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
    dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'; dsm_fn = lambda t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif'
    dsm_tiles = {t: dsm_fn(t) for t in train_t + val_t}
    train_ds = Window256DatasetBO(img_dir, gt_dir, train_t, '.tif', '.tif', dsm_tiles, True)
    val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, '.tif', '.tif', dsm_tiles, False)
    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} samples, {len(train_ldr)} batches")

    print("\nBuilding model..."); sam3 = load_sam3()
    model = VPT_MFNetDecoder(sam3, adapter_bottleneck=32, num_classes=5, use_dsm=True, dropout=0.1).cuda(); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs); scaler = torch.amp.GradScaler('cuda')
    start_epoch, best_v = 1, 0.0; hist = {'loss': [], 'lr': [], 'miou': []}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); ep_loss, n_batches = 0.0, 0; t0 = time.time()
        for bi, (images, dsm, labels, boundary, objects) in enumerate(train_ldr):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            boundary, objects = boundary.to(device), objects.to(device)
            logits = model(images, dsm); logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
            # SAM_RS weights: λ_boundary=0.01, λ_object=0.01
            loss = structure_loss(logits, labels)
            loss = loss + 0.01 * boundary_loss(logits, boundary)
            loss = loss + 0.01 * object_loss(logits, objects)
            # Car weighting: ×3 extra BCE for car class (only 1.1% pixels)
            car_mask = (labels == 4).float()
            if car_mask.sum() > 10:  # at least 10 car pixels
                car_logits = logits[:, 4]  # car channel (B, H, W)
                car_loss = F.binary_cross_entropy_with_logits(
                    car_logits, car_mask, reduction='mean')
                loss = loss + 3.0 * car_loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            ep_loss += loss.item(); n_batches += 1
            if bi % 100 == 0: print(f"  E{epoch:3d}/{args.epochs} B{bi:4d}/{len(train_ldr)} loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e}")
        sch.step(); avg_loss = ep_loss / max(n_batches, 1)
        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val: aAcc, mIoU, per_class = validate(model, val_ldr, device)
        else: mIoU = hist['miou'][-1] if hist['miou'] else 0.0; per_class = []
        hist['loss'].append(avg_loss); hist['lr'].append(sch.get_last_lr()[0]); hist['miou'].append(mIoU)
        if do_val: print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% (best={best_v:.1f}) [{time.time()-t0:.0f}s]\n    per-class: {[f'{x:.1f}' for x in per_class]}")
        if do_val and mIoU > best_v: best_v = mIoU; torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_v': best_v, 'hist': hist}, os.path.join(out_dir, 'best_model.pt'))
    json.dump(hist, open(os.path.join(out_dir, 'history.json'), 'w'))
    del sam3, model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Best mIoU={best_v:.1f}%\nOutput: {out_dir}")

if __name__ == '__main__': main()
