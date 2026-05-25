#!/usr/bin/env python3
"""
MFNet-style 256² sliding window training.

Key difference from previous training:
- 256² non-overlapping grid patches instead of 512² random crops
- 3.3× more training samples (3420 vs 960 for Vaihingen)
- Resize 256→1008 for SAM3 ViT, then resize output back to 256
- Matches MFNet's training protocol

用法:
  cd /root/Mynet/RS-SAM3-p3r
  python train_256.py --model dsm --epochs 20 --batch 8
"""

import os, sys, json, time, gc, argparse, random
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p3r')

from structure_loss import structure_loss
from dataset_adapter import (_rgb_to_class, NUM_CLASSES, IGNORE_INDEX,
                             VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                             POTSDAM_TRAIN, POTSDAM_VAL)

# ── 256² Window Dataset ──────────────────────────────────

class Window256Dataset(torch.utils.data.Dataset):
    """Pre-extract 256² patches with stride=128 from tiles (MFNet protocol)."""

    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                 is_train=True, stride=128):
        self.is_train = is_train
        self.samples = []
        self.labels = []

        for tile in tiles:
            ip = f'{img_dir}/{tile}{img_suffix}'
            gp = f'{gt_dir}/{tile}{gt_suffix}'
            if not os.path.exists(ip):
                continue
            img = np.array(Image.open(ip).convert('RGB'))
            gt_rgb = np.array(Image.open(gp).convert('RGB'))
            gt = _rgb_to_class(gt_rgb)

            h, w = img.shape[:2]
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y + 256, h), min(x + 256, w)
                    ph, pw = y2 - y, x2 - x
                    if ph < 128 or pw < 128:
                        continue
                    patch = img[y:y2, x:x2]
                    label = gt[y:y2, x:x2]
                    # Pad to exactly 256×256 if edge fragment
                    if ph < 256 or pw < 256:
                        patch = np.pad(patch, ((0, 256-ph), (0, 256-pw), (0, 0)), mode='reflect')
                        label = np.pad(label, ((0, 256-ph), (0, 256-pw)), mode='constant', constant_values=255)
                    # Skip patches that are mostly ignore
                    if (label == 255).mean() > 0.5:
                        continue
                    self.samples.append(patch)
                    self.labels.append(label)

        print(f"  {len(self.samples)} patches from {len(tiles)} tiles "
              f"({'train' if is_train else 'val'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        lbl = self.labels[idx]

        if self.is_train:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                lbl = np.fliplr(lbl).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                lbl = np.flipud(lbl).copy()
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                lbl = np.rot90(lbl, k).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        lbl_t = torch.from_numpy(lbl).long()
        return img_t, lbl_t


class Window256DatasetDSM(Window256Dataset):
    """256² window dataset with DSM patches."""

    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                 dsm_tiles, is_train=True, stride=128):
        # Load DSMs for all tiles
        dsm_data = {}
        for tile, dsm_path in dsm_tiles.items():
            if os.path.exists(dsm_path):
                d = np.array(Image.open(dsm_path)).astype(np.float32)
                d = (d - d.min()) / max(d.max() - d.min(), 1e-8)
                dsm_data[tile] = d
            else:
                dsm_data[tile] = None

        super().__init__(img_dir, gt_dir, tiles, img_suffix, gt_suffix, is_train, stride)

        # Re-extract DSM patches matching the image patches
        self.dsms = []
        for tile in tiles:
            ip = f'{img_dir}/{tile}{img_suffix}'
            if not os.path.exists(ip):
                continue
            dsm = dsm_data.get(tile)
            h, w = dsm.shape if dsm is not None else (0, 0)
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y + 256, h), min(x + 256, w)
                    ph, pw = y2 - y, x2 - x
                    if ph < 128 or pw < 128:
                        continue
                    dsm_patch = dsm[y:y2, x:x2] if dsm is not None else np.zeros((ph, pw), dtype=np.float32)
                    if ph < 256 or pw < 256:
                        dsm_patch = np.pad(dsm_patch, ((0, 256-ph), (0, 256-pw)), mode='reflect')
                    self.dsms.append(dsm_patch)

        # Trim DSM list to match samples (may differ due to filtering)
        self.dsms = self.dsms[:len(self.samples)]
        print(f"  {len(self.dsms)} DSM patches loaded")

    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        dsm = self.dsms[idx].copy()
        # Re-apply same augmentations to DSM (parent already augmented img/lbl)
        if self.is_train:
            rng_state = random.getstate()
            random.seed(idx)
            if random.random() < 0.5:
                dsm = np.fliplr(dsm).copy()
            if random.random() < 0.5:
                dsm = np.flipud(dsm).copy()
            random.setstate(rng_state)
        dsm_t = torch.from_numpy(dsm).float()
        return img, dsm_t, lbl


# ── Training ──────────────────────────────────────────────

def load_sam3(device='cuda'):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    return model.cuda()


@torch.no_grad()
def validate(model, loader, device, use_dsm):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct, total = 0, 0
    for batch in loader:
        if use_dsm:
            images, dsm, labels = batch
            dsm = dsm.to(device)
        else:
            images, labels = batch
            dsm = None
        images, labels = images.to(device), labels.to(device)
        logits = model(images, dsm) if use_dsm else model(images)
        logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)
        mask = (labels != IGNORE_INDEX)
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c
            inter[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()
    aAcc = correct / max(total, 1) * 100
    pc_iou = [(inter[c] / max(union[c], 1) * 100).item() for c in range(NUM_CLASSES)]
    pc_oa = [((inter[c] + total - union[c]) / max(total, 1) * 100).item() for c in range(NUM_CLASSES)]
    return aAcc, np.mean(pc_iou), pc_iou, pc_oa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dsm', choices=['rgb', 'dsm'])
    parser.add_argument('--dataset', default='vaihingen')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--output', default='/root/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan3_256win_{args.model}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    use_dsm = (args.model == 'dsm')
    print(f"256² Window Training (MFNet protocol)")
    print(f"  Model: {args.model}, Dataset: {args.dataset}")
    print(f"  Output: {out_dir}")

    # Data
    print("\nLoading data...")
    if args.dataset == 'vaihingen':
        train_t, val_t = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        isuf, gsuf = '.tif', '.tif'
    else:
        train_t, val_t = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        isuf, gsuf = '_RGB.tif', '_label.tif'

    if use_dsm:
        # Build DSM path mapping
        if args.dataset == 'vaihingen':
            dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
            dsm_tiles = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif'
                         for t in train_t + val_t}
        else:
            dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
            dsm_tiles = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_","")}.tif'
                         for t in train_t + val_t}
        train_ds = Window256DatasetDSM(img_dir, gt_dir, train_t, isuf, gsuf, dsm_tiles, True)
        val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, isuf, gsuf, dsm_tiles, False)
    else:
        train_ds = Window256Dataset(img_dir, gt_dir, train_t, isuf, gsuf, True)
        val_ds = Window256Dataset(img_dir, gt_dir, val_t, isuf, gsuf, False)

    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                            num_workers=0, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Train batches: {len(train_ldr)} ({len(train_ds)} samples)")

    # Model
    print("\nBuilding model...")
    sam3 = load_sam3(device)
    if use_dsm:
        from adapter_unet import AdapterSAM3UNetFormerDSM
        model = AdapterSAM3UNetFormerDSM(sam3, adapter_bottleneck=32, num_classes=NUM_CLASSES,
                                         dropout=0.1).cuda()
    else:
        from adapter_unet import AdapterSAM3UNetFormer
        model = AdapterSAM3UNetFormer(sam3, adapter_bottleneck=32, num_classes=NUM_CLASSES,
                                      dropout=0.1).cuda()
    model.resolution = 1008
    model.train()
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_v = 1, 0.0
    hist = {'loss': [], 'lr': [], 'miou': []}

    print(f"\n{'='*60}")
    print(f"Training {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        for bi, batch in enumerate(train_ldr):
            if use_dsm:
                images, dsm, labels = batch
                dsm = dsm.to(device)
            else:
                images, labels = batch
                dsm = None
            images, labels = images.to(device), labels.to(device)

            # Resize 256→1008 for SAM3 ViT, model handles this internally
            logits = model(images, dsm) if use_dsm else model(images)
            # Resize output back to 256 for loss
            logits = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = structure_loss(logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item(); n_batches += 1
            if bi % 50 == 0:
                print(f"  E{epoch:3d}/{args.epochs} B{bi:4d}/{len(train_ldr)} "
                      f"loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e}")

        sch.step()
        avg_loss = ep_loss / max(n_batches, 1)

        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            aAcc, mIoU, per_class, per_class_oa = validate(model, val_ldr, device, use_dsm)
        else:
            mIoU = hist['miou'][-1] if hist['miou'] else 0.0
            per_class = []
            per_class_oa = []

        hist['loss'].append(avg_loss); hist['lr'].append(sch.get_last_lr()[0])
        hist['miou'].append(mIoU)
        eta = time.time() - t0
        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% (best={best_v:.1f}) [{eta:.0f}s]")
            print(f"    per-class IoU: {[f'{x:.1f}' for x in per_class]}")
            print(f"    per-class OA : {[f'{x:.1f}' for x in per_class_oa]}")
        else:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} (skip val) [{eta:.0f}s]")

        if do_val and mIoU > best_v:
            best_v = mIoU
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
                        'best_v': best_v, 'hist': hist, 'args': vars(args)},
                       os.path.join(out_dir, 'best_model.pt'))
            print(f"    ✓ Saved best (mIoU={best_v:.1f}%)")
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
                        'best_v': best_v, 'hist': hist, 'args': vars(args)},
                       os.path.join(out_dir, f'checkpoint_epoch{epoch:03d}.pt'))

    json.dump(hist, open(os.path.join(out_dir, 'history.json'), 'w'))
    del sam3, model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Best mIoU={best_v:.1f}%\nOutput: {out_dir}")


if __name__ == '__main__':
    main()
