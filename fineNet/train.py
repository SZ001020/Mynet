"""
Phase 3: SAM 3 遥感微调训练脚本

支持三种策略: partial (冻结backbone), full (全量), lora (LoRA注入)
输出: checkpoint, loss curve, per-class IoU
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

sys.path.insert(0, '/root/Mynet/SegEarth-OV-3-main')
sys.path.insert(0, '/root/Mynet/sam3-main')

from finetune_model import FineTunedSAM3, count_trainable_params
from finetune_dataset import create_dataloaders, NUM_CLASSES, IGNORE_INDEX


def compute_iou(pred, target, num_classes):
    """Compute per-class and mean IoU."""
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        iou = intersection / union if union > 0 else float('nan')
        ious.append(iou)
    mean_iou = np.nanmean(ious) * 100
    return mean_iou, [iou * 100 for iou in ious]


@torch.no_grad()
def validate(model, val_loader, num_classes, device, dataset_name=''):
    """验证函数：逐图计算，累积混淆矩阵."""
    model.eval()
    total_intersect = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_correct = 0
    total_pixels = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)

        preds = logits.argmax(1)
        mask = (labels != IGNORE_INDEX)
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_pixels += mask.sum().item()

        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (labels == c)
            total_intersect[c] += (pred_c & target_c).sum()
            total_union[c] += (pred_c | target_c).sum()
            class_correct[c] += (pred_c[mask] & target_c[mask]).sum()
            class_total[c] += target_c[mask].sum()

    aAcc = total_correct / max(total_pixels, 1) * 100

    per_class_iou = []
    for c in range(num_classes):
        iou = total_intersect[c].item() / max(total_union[c].item(), 1) * 100
        per_class_iou.append(iou)
    miou = np.mean(per_class_iou) if per_class_iou else 0.0

    per_class_acc = []
    for c in range(num_classes):
        acc = class_correct[c].item() / max(class_total[c].item(), 1) * 100
        per_class_acc.append(acc)
    mAcc = np.mean(per_class_acc) if per_class_acc else 0.0

    model.train()
    return aAcc, miou, mAcc, per_class_iou


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, total_epochs):
    """训练一个 epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch}/{total_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}",
                  end='\r')

    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['partial', 'full', 'lora'],
                        help='Fine-tuning mode')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='Random crop size')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank (only for lora mode)')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['vaihingen', 'potsdam', None],
                        help='Train on a single dataset (default: combined)')
    parser.add_argument('--output-dir', type=str,
                        default='/root/Mynet/autodl-tmp/runs',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda')
    ds_tag = args.dataset or 'combined'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'phase3_{args.mode}_{ds_tag}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 3: SAM 3 Fine-tuning — Mode: {args.mode.upper()}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Crop: {args.crop_size}")

    # Create model
    print(f"\nBuilding SAM 3 model ({args.mode} mode)...")
    model = FineTunedSAM3(
        num_classes=NUM_CLASSES,
        mode=args.mode,
        lora_rank=args.lora_rank,
        device=device,
    )
    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")

    # Data
    print(f"\nLoading datasets (dataset={args.dataset or 'combined'})...")
    train_loader, v_val_loader, p_val_loader = create_dataloaders(
        batch_size=args.batch_size, crop_size=args.crop_size, dataset_name=args.dataset)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # Training loop
    best_v_miou = 0.0
    best_p_miou = 0.0
    history = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"Training started ({args.epochs} epochs)")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs)
        scheduler.step()

        # Validate
        v_aacc, v_miou, v_macc, v_per_cls = validate(model, v_val_loader, NUM_CLASSES, device, 'Vaihingen')
        p_aacc, p_miou, p_macc, p_per_cls = validate(model, p_val_loader, NUM_CLASSES, device, 'Potsdam')

        elapsed = time.time() - epoch_start

        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['v_miou'].append(v_miou)
        history['p_miou'].append(p_miou)
        history['v_aacc'].append(v_aacc)
        history['p_aacc'].append(p_aacc)

        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"V-mIoU: {v_miou:.1f}% | P-mIoU: {p_miou:.1f}% | "
              f"V-aAcc: {v_aacc:.1f}% | P-aAcc: {p_aacc:.1f}% | {elapsed:.0f}s")

        # Save best
        if args.dataset == 'potsdam':
            if p_miou > best_p_miou:
                best_p_miou = p_miou
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'v_miou': v_miou, 'p_miou': p_miou, 'dataset': args.dataset,
                    'mode': args.mode, 'per_class_iou': p_per_cls,
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"    → Best checkpoint (mIoU={p_miou:.1f}%)")
        else:
            if v_miou > best_v_miou:
                best_v_miou = v_miou
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'v_miou': v_miou, 'p_miou': p_miou, 'dataset': args.dataset,
                    'mode': args.mode, 'per_class_iou': v_per_cls,
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"    → Best checkpoint (mIoU={v_miou:.1f}%)")

            if p_miou > best_p_miou:
                best_p_miou = p_miou
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'v_miou': v_miou, 'p_miou': p_miou, 'dataset': args.dataset,
                    'mode': args.mode, 'per_class_iou': p_per_cls,
                }, os.path.join(output_dir, 'best_model_potsdam.pt'))

    # Final save
    torch.save({
        'epoch': args.epochs, 'model_state_dict': model.state_dict(),
        'history': dict(history),
        'mode': args.mode, 'best_v_miou': best_v_miou, 'best_p_miou': best_p_miou,
    }, os.path.join(output_dir, 'final_model.pt'))

    # Save training history
    json.dump(dict(history), open(os.path.join(output_dir, 'training_history.json'), 'w'), indent=2)

    # Save config
    config = {
        'mode': args.mode, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'crop_size': args.crop_size, 'lora_rank': args.lora_rank,
        'trainable_params': trainable, 'total_params': total,
        'best_v_miou': best_v_miou, 'best_p_miou': best_p_miou,
        'timestamp': timestamp,
    }
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best Vaihingen mIoU: {best_v_miou:.1f}%")
    print(f"  Best Potsdam mIoU: {best_p_miou:.1f}%")
    print(f"  Output: {output_dir}")

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
