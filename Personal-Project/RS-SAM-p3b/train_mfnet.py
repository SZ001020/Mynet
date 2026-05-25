#!/usr/bin/env python3
"""
MFNet-style deep DSM fusion training: DSM goes through SAM3 ViT (shared weights).

用法:
  cd /root/Mynet/RS-SAM-p3b
  python train_mfnet.py --dataset vaihingen --rank 8 --epochs 20 --batch 2
"""

import os, sys, json, time, gc, argparse
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM-p3b')

from lora_mfnet import LoRASAM3MFNet
from structure_loss import structure_loss
from dataset_adapter import (ISPRSTrainDataset, ISPRSTrainDatasetDSM, DSM_PATTERNS,
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


@torch.no_grad()
def validate(model, loader, device, use_dsm):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct, total = 0, 0
    for batch in loader:
        if len(batch) == 3:
            images, dsm, labels = batch
            dsm = dsm.to(device) if use_dsm else None
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
    return aAcc, np.mean(pc_iou), pc_iou


def main():
    parser = argparse.ArgumentParser(description='MFNet-style LoRA-SAM3 + DSM')
    parser.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--output', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan3_mfnet_r{args.rank}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    use_dsm = True
    print(f"MFNet-style LoRA-SAM3 + Deep DSM Fusion")
    print(f"  Dataset: {args.dataset}, Rank: {args.rank}, Epochs: {args.epochs}")
    print(f"  Output: {out_dir}")

    # Data with DSM
    print("\nLoading data...")
    dsm_dir, dsm_pat, stem_fn = DSM_PATTERNS[args.dataset]
    if args.dataset == 'vaihingen':
        td, vd = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        isuf, gsuf = '.tif', '.tif'
    else:
        td, vd = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        isuf, gsuf = '_RGB.tif', '_label.tif'
    train_ds = ISPRSTrainDatasetDSM(img_dir, gt_dir, td, isuf, gsuf,
                                    dsm_dir, dsm_pat, stem_fn, args.crop, True)
    val_ds = ISPRSTrainDatasetDSM(img_dir, gt_dir, vd, isuf, gsuf,
                                  dsm_dir, dsm_pat, stem_fn, args.crop, False)
    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                            num_workers=0, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Train batches: {len(train_ldr)}")

    # Model
    print("\nBuilding model...")
    sam3 = load_sam3(device)
    model = LoRASAM3MFNet(sam3, lora_rank=args.rank, lora_alpha=args.rank*2,
                          num_classes=NUM_CLASSES, dropout=0.1).cuda()
    model.resolution = 1008
    model.train()

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

        for bi, (images, dsm, labels) in enumerate(train_ldr):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)

            logits = model(images, dsm)
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
            aAcc, mIoU, per_class = validate(model, val_ldr, device, use_dsm)
        else:
            mIoU = hist['miou'][-1] if hist['miou'] else 0.0
            per_class = []

        hist['loss'].append(avg_loss); hist['lr'].append(sch.get_last_lr()[0])
        hist['miou'].append(mIoU)

        eta = time.time() - t0
        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% (best={best_v:.1f}) [{eta:.0f}s]")
            print(f"    per-class: {[f'{x:.1f}' for x in per_class]}")
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
