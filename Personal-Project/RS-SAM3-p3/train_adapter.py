#!/usr/bin/env python3
"""
Plan3 Route A: Adapter-ViT + UNet Decoder 训练脚本

架构:
  SAM3 ViTDet (32 AdapterBlocks, trainable prompt_learn ~2M params)
    → FPN (frozen, 3 scales: 288², 144², 72²)
    → UNet Decoder (trainable)
    → 5-channel per-class logits

Loss: structure_loss = 边缘加权 BCE + 加权 IoU
数据: Vaihingen (12 tiles) + Potsdam (18 tiles), 512² random crops
优化: AdamW lr=1e-4, CosineAnnealing, 50 epochs

用法:
  cd /root/Mynet/RS-SAM3-p3
  python train_adapter.py [--epochs 50] [--batch 8] [--lr 1e-4]
"""

import os, sys, json, time, gc, random, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────
SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)

# Import local modules
from structure_loss import structure_loss
from dataset_adapter import create_dataloaders, NUM_CLASSES, IGNORE_INDEX


# ============================================================
# Model Builder
# ============================================================

def load_sam3_model(device='cuda'):
    """Load SAM3 from SegEarth checkpoint."""
    print("Loading SAM3 (SegEarth)...")
    _prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt',
        device=device)
    os.chdir(_prev)

    # Force all params to CUDA (activation checkpointing may leave some on CPU)
    model = model.cuda()
    for p in model.parameters():
        p.data = p.data.cuda()
    return model


def build_model(sam3_model, adapter_bottleneck=32, num_classes=NUM_CLASSES):
    """Build the Adapter-SAM3-UNet model."""
    from adapter_unet import AdapterSAM3UNet
    model = AdapterSAM3UNet(sam3_model, adapter_bottleneck=adapter_bottleneck,
                             num_classes=num_classes)
    return model.cuda()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, device):
    """Per-class IoU evaluation.

    Returns: (aAcc, mIoU, per_class_IoU_list)
    """
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        # Resize logits to match label size
        logits = F.interpolate(logits, labels.shape[-2:],
                               mode='bilinear', align_corners=False)
        pred = logits.argmax(1)

        mask = (labels != IGNORE_INDEX)
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()

        for c in range(NUM_CLASSES):
            pc = (pred == c)
            lc = (labels == c)
            inter[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()

    aAcc = correct / max(total, 1) * 100
    per_class_iou = [(inter[c] / max(union[c], 1) * 100).item() for c in range(NUM_CLASSES)]
    per_class_oa = [((inter[c] + total - union[c]) / max(total, 1) * 100).item() for c in range(NUM_CLASSES)]
    mIoU = np.mean(per_class_iou)
    return aAcc, mIoU, per_class_iou, per_class_oa


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Plan3 Route A: Adapter-ViT + UNet Decoder')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='Adapter bottleneck dim (default 32)')
    parser.add_argument('--resolution', type=int, default=1008,
                        help='Backbone input resolution (SAM3 native=1008, hardcoded RoPE)')
    parser.add_argument('--val-every', type=int, default=5,
                        help='Validate every N epochs (default 5, save time)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0=safest, avoids CUDA fork issues)')
    parser.add_argument('--dataset', default=None,
                        choices=['vaihingen', 'potsdam', None],
                        help='Train on single dataset (default: both)')
    parser.add_argument('--output', default='/root/Mynet/autodl-tmp/runs')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan3_adapter_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plan3 Route A: Adapter-ViT + UNet Decoder")
    print(f"  Output: {out_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    print(f"  Resolution: {args.resolution} (SAM3 native=1008), Val every: {args.val_every}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Adapter bottleneck: {args.bottleneck}")

    # ── Data ──
    print("\nLoading data...")
    train_ldr, v_ldr, p_ldr = create_dataloaders(
        batch_size=args.batch, crop_size=args.crop, dataset_name=args.dataset,
        num_workers=args.num_workers)
    print(f"  Train batches: {len(train_ldr)}")

    # ── Model ──
    print("\nBuilding model...")
    sam3 = load_sam3_model(device)
    model = build_model(sam3, adapter_bottleneck=args.bottleneck, num_classes=NUM_CLASSES)
    model.resolution = args.resolution
    model.train()

    # Ensure backbone operates in no_grad for all non-adapter params
    # This is handled by AdapterBlock and inject_adapters setting requires_grad=False

    # ── Optimizer ──
    # Only train adapter prompt_learn + decoder params
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_v, best_p = 0.0, 0.0
    hist = {'loss': [], 'lr': [], 'v_miou': [], 'p_miou': [],
            'v_aAcc': [], 'p_aAcc': []}

    # ── Resume ──
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        opt.load_state_dict(ckpt['optimizer'])
        sch.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_v = ckpt.get('best_v', 0.0)
        best_p = ckpt.get('best_p', 0.0)
        if 'hist' in ckpt:
            hist = ckpt['hist']

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"Training {args.epochs} epochs (starting from {start_epoch})...")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        for bi, (images, labels) in enumerate(train_ldr):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            logits = F.interpolate(logits, labels.shape[-2:],
                                   mode='bilinear', align_corners=False)

            # Structure loss (edge-weighted BCE + IoU)
            loss = structure_loss(logits, labels)

            # Backward
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
        eta = time.time() - t0

        # ── Validation (every --val-every epochs, or on epoch 1 for baseline) ──
        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            v_a, v_m, v_pc, v_pc_oa = validate(model, v_ldr, device)
            p_a, p_m, p_pc, p_pc_oa = validate(model, p_ldr, device)
        else:
            # Track last known values (don't spend time on validation)
            v_m = hist['v_miou'][-1] if hist['v_miou'] else 0.0
            p_m = hist['p_miou'][-1] if hist['p_miou'] else 0.0

        hist['loss'].append(avg_loss)
        hist['lr'].append(sch.get_last_lr()[0])
        hist['v_miou'].append(v_m)
        hist['p_miou'].append(p_m)

        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} "
                  f"V={v_m:.1f}% (best={best_v:.1f}) P={p_m:.1f}% (best={best_p:.1f}) "
                  f"[{eta:.0f}s]")
            print(f"    V per-class IoU: {[f'{x:.1f}' for x in v_pc]}")
            print(f"    V per-class OA : {[f'{x:.1f}' for x in v_pc_oa]}")
            print(f"    P per-class IoU: {[f'{x:.1f}' for x in p_pc]}")
            print(f"    P per-class OA : {[f'{x:.1f}' for x in p_pc_oa]}")
            hist['v_aAcc'].append(v_a)
            hist['p_aAcc'].append(p_a)
        else:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} (skip val) [{eta:.0f}s]")

        # ── Save best ──
        is_best = False
        if do_val:
            if v_m > best_v:
                best_v = v_m
                is_best = True
            if p_m > best_p:
                best_p = p_m

        if is_best or epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sch.state_dict(),
                'best_v': best_v,
                'best_p': best_p,
                'hist': hist,
                'args': vars(args),
            }, os.path.join(out_dir, 'best_model.pt' if is_best else f'checkpoint_epoch{epoch:03d}.pt'))
            if is_best:
                print(f"    ✓ Saved best model (V={best_v:.1f}%, P={best_p:.1f}%)")

    # ── Done ──
    json.dump(hist, open(os.path.join(out_dir, 'history.json'), 'w'))
    del sam3, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{'='*60}")
    print(f"Done! Best: V={best_v:.1f}% P={best_p:.1f}%")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
