#!/usr/bin/env python3
"""
Plan3 Route A+DSM: RGB + DSM Dual-Stream 训练

架构:
  SAM3 ViT + Adapter (RGB) → FPN features (3 scales)
  DSM CNN Encoder           → DSM features (3 scales)
  → concat + 1×1 fusion → UNet decoder → 5-class logits

用法:
  cd /root/Mynet/RS-SAM3-p3
  python train_dual.py --dataset vaihingen --epochs 20 --batch 4
"""

import os, sys, json, time, gc, argparse
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p3')

from adapter_unet import AdapterSAM3UNetDSM
from structure_loss import structure_loss
from dataset_adapter import (ISPRSTrainDatasetDSM, DSM_PATTERNS,
                             VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                             POTSDAM_TRAIN, POTSDAM_VAL,
                             NUM_CLASSES, IGNORE_INDEX)


def load_sam3(device='cuda'):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    return model.cuda()


def create_dsm_dataloaders(dataset_name, batch_size=4, crop_size=512, num_workers=0):
    """Create dataloaders with DSM."""
    dsm_dir, dsm_pat, stem_fn = DSM_PATTERNS[dataset_name]

    if dataset_name == 'vaihingen':
        train_tiles, val_tiles = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        img_suf, gt_suf = '.tif', '.tif'
    else:
        train_tiles, val_tiles = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'

    train_ds = ISPRSTrainDatasetDSM(
        img_dir, gt_dir, train_tiles, img_suf, gt_suf,
        dsm_dir, dsm_pat, stem_fn, crop_size, is_train=True)
    val_ds = ISPRSTrainDatasetDSM(
        img_dir, gt_dir, val_tiles, img_suf, gt_suf,
        dsm_dir, dsm_pat, stem_fn, crop_size, is_train=False)

    train_ldr = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return train_ldr, val_ldr


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct, total = 0, 0

    for images, dsm, labels in loader:
        images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
        logits = model(images, dsm)
        logits = F.interpolate(logits, labels.shape[-2:],
                               mode='bilinear', align_corners=False)
        pred = logits.argmax(1)
        mask = (labels != IGNORE_INDEX)
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c
            inter[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()

    aAcc = correct / max(total, 1) * 100
    per_class_iou = [(inter[c] / max(union[c], 1) * 100).item() for c in range(NUM_CLASSES)]
    per_class_oa = [((inter[c] + total - union[c]) / max(total, 1) * 100).item() for c in range(NUM_CLASSES)]
    mIoU = np.mean(per_class_iou)
    return aAcc, mIoU, per_class_iou, per_class_oa


def main():
    parser = argparse.ArgumentParser(description='Plan3 Route A+DSM: RGB + DSM Dual-Stream')
    parser.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output', default='/root/Mynet/autodl-tmp/runs')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan3_dual_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plan3 Route A+DSM: RGB + DSM Dual-Stream")
    print(f"  Dataset: {args.dataset}, Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"  Output: {out_dir}")

    # Data
    print("\nLoading data...")
    train_ldr, val_ldr = create_dsm_dataloaders(
        args.dataset, args.batch, args.crop, args.num_workers)
    print(f"  Train batches: {len(train_ldr)}")

    # Model
    print("\nBuilding model...")
    sam3 = load_sam3(device)
    model = AdapterSAM3UNetDSM(sam3, adapter_bottleneck=32, num_classes=NUM_CLASSES,
                               dropout=0.1).cuda()
    model.resolution = 1008
    model.train()
    print(f"  Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_v, best_p = 1, 0.0, 0.0
    hist = {'loss': [], 'lr': [], 'miou': []}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        opt.load_state_dict(ckpt['optimizer'])
        sch.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_v = ckpt.get('best_v', 0.0)
        hist = ckpt.get('hist', hist)

    print(f"\n{'='*60}")
    print(f"Training {args.epochs} epochs (starting from {start_epoch})...")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        for bi, (images, dsm, labels) in enumerate(train_ldr):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)

            logits = model(images, dsm)
            logits = F.interpolate(logits, labels.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss = structure_loss(logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ep_loss += loss.item()
            n_batches += 1

            if bi % 50 == 0:
                print(f"  E{epoch:3d}/{args.epochs} B{bi:4d}/{len(train_ldr)} "
                      f"loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e}")

        sch.step()
        avg_loss = ep_loss / max(n_batches, 1)

        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            aAcc, mIoU, per_class, per_class_oa = validate(model, val_ldr, device)
        else:
            mIoU = hist['miou'][-1] if hist['miou'] else 0.0

        hist['loss'].append(avg_loss)
        hist['lr'].append(sch.get_last_lr()[0])
        hist['miou'].append(mIoU)

        eta = time.time() - t0
        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% "
                  f"(best={best_v:.1f}) [{eta:.0f}s]")
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
            print(f"    ✓ Saved best model (mIoU={best_v:.1f}%)")

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
