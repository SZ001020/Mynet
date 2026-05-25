#!/usr/bin/env python3
"""
Plan4: Full training of SAM3 ViTDet backbone (MFNet recipe).

Key differences from frozen training:
1. All ViT params trainable (no VPT adapter needed)
2. SGD + momentum=0.9 + weight_decay=5e-4 (matching MFNet)
3. Linear warmup 2 epochs → CosineAnnealing 50 epochs
4. Deletes language_backbone to save VRAM

用法:
  cd /root/Mynet/RS-SAM3-p4
  CUDA_VISIBLE_DEVICES=0 python train_full.py --dataset vaihingen --epochs 50 --batch 4
"""

import os, sys, json, time, gc, argparse, random, math
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p4')
sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM-p3b')

from structure_loss import structure_loss
from mfnet_decoder import MFNetDecoder, Pyramid4Scale, SEFusion
from dataset_adapter import (_rgb_to_class, NUM_CLASSES, IGNORE_INDEX,
                             VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                             POTSDAM_TRAIN, POTSDAM_VAL)
from train_256 import Window256Dataset, Window256DatasetDSM


class SAM3FullTrain(nn.Module):
    """SAM3 ViTDet full training model (no VPT, no frozen layers)."""

    def __init__(self, sam3_model, num_classes=5, use_dsm=True, dropout=0.1):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.use_dsm = use_dsm

        # All ViT params trainable
        vb = self.backbone.vision_backbone
        for p in vb.parameters():
            p.requires_grad = True
        n_vit = sum(p.numel() for p in vb.parameters())
        print(f"  ViT trainable: {n_vit:,} ({n_vit*4/1e9:.1f}GB)")

        # Delete language backbone (not used, saves 1.4GB)
        if hasattr(self.backbone, 'language_backbone'):
            del self.backbone.language_backbone
            print(f"  Deleted language_backbone (saved ~1.4GB)")

        # 4-scale pyramid (ViT output → 4 MFNet scales)
        self.pyramid_rgb = Pyramid4Scale(256)

        # DSM encoder + SEFusion
        if use_dsm:
            from adapter_vit import DSMEncoderDeep
            self.dsm_encoder = DSMEncoderDeep(dsm_dim=128)
            self.pyramid_dsm = Pyramid4Scale(128)
            self.fusion1 = SEFusion(256)
            self.fusion2 = SEFusion(256)
            self.fusion3 = SEFusion(256)
            self.fusion4 = SEFusion(256)
            self.dsm_proj = nn.ModuleList([nn.Conv2d(128, 256, 1) for _ in range(4)])

        # MFNet Decoder
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64,
                                    dropout=dropout)
        n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total trainable: {n_total:,}")

    def forward(self, images, dsm=None):
        # Normalize + resize for SAM3 ViT
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != 1008 or w != 1008:
            images_norm = F.interpolate(images_norm, (1008, 1008),
                                        mode='bilinear', align_corners=False)

        out = self.backbone.forward_image(images_norm)
        fpn = out['backbone_fpn']
        vit_feat = fpn[-1].clone()  # deepest FPN scale (72²)

        scales_rgb = self.pyramid_rgb(vit_feat)

        if dsm is not None and self.use_dsm:
            dsm_out = self.dsm_encoder(dsm)
            dsm_feat = dsm_out['features']
            if dsm_feat.shape[-2:] != vit_feat.shape[-2:]:
                dsm_feat = F.interpolate(dsm_feat, vit_feat.shape[-2:],
                                         mode='bilinear', align_corners=False)
            scales_dsm = self.pyramid_dsm(dsm_feat)
            fusions = [self.fusion1, self.fusion2, self.fusion3, self.fusion4]
            feats = []
            for i in range(4):
                dsm_p = self.dsm_proj[i](scales_dsm[i])
                if dsm_p.shape[-2:] != scales_rgb[i].shape[-2:]:
                    dsm_p = F.interpolate(dsm_p, scales_rgb[i].shape[-2:],
                                          mode='bilinear', align_corners=False)
                feats.append(fusions[i](scales_rgb[i], dsm_p))
        else:
            feats = scales_rgb

        return self.decoder(feats)


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
            images, dsm, labels = batch; dsm = dsm.to(device)
        else:
            images, labels = batch; dsm = None
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
    parser = argparse.ArgumentParser(description='Plan4: Full SAM3 ViT training')
    parser.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--output', default='/root/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan4_full_{args.dataset}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    use_dsm = True
    print(f"Plan4 Full Training: SAM3 ViTDet + MFNet Decoder + DSM")
    print(f"  Dataset: {args.dataset}, Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"  LR: {args.lr} (SGD+momentum), Warmup: {args.warmup} epochs")
    print(f"  Output: {out_dir}")

    # Data
    print("\nLoading data...")
    if args.dataset == 'vaihingen':
        train_t, val_t = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        isuf, gsuf = '.tif', '.tif'
        dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
        dsm_pat = 'dsm_09cm_matching_area{}.tif'
        dsm_stem = lambda t: t.replace('top_mosaic_09cm_area', '')
    else:
        train_t, val_t = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        isuf, gsuf = '_RGB.tif', '_label.tif'
        dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
        dsm_pat = 'dsm_potsdam_{}.tif'
        dsm_stem = lambda t: t.replace('top_potsdam_', '')

    dsm_tiles = {t: f'{dsm_dir}/{dsm_pat.format(dsm_stem(t))}' for t in train_t + val_t}
    train_ds = Window256DatasetDSM(img_dir, gt_dir, train_t, isuf, gsuf, dsm_tiles, True)
    val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, isuf, gsuf, dsm_tiles, False)
    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                            num_workers=0, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} samples, {len(train_ldr)} batches")

    # Model
    print("\nBuilding model...")
    sam3 = load_sam3(device)
    model = SAM3FullTrain(sam3, num_classes=NUM_CLASSES, use_dsm=use_dsm, dropout=0.1).cuda()
    model.train()

    # SGD + momentum (MFNet recipe)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # CosineAnnealing after warmup
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - args.warmup)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_v = 1, 0.0
    hist = {'loss': [], 'lr': [], 'miou': []}

    print(f"\n{'='*60}")
    print(f"Training {args.epochs} epochs (warmup={args.warmup}, SGD lr={args.lr})...")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Warmup: linear increase from lr/100 to lr
        if epoch <= args.warmup:
            warmup_lr = args.lr * (epoch / args.warmup)
            for pg in opt.param_groups:
                pg['lr'] = warmup_lr

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
            if bi % 100 == 0:
                print(f"  E{epoch:3d}/{args.epochs} B{bi:4d}/{len(train_ldr)} "
                      f"loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")

        avg_loss = ep_loss / max(n_batches, 1)

        # CosineAnnealing after warmup
        if epoch > args.warmup:
            sch.step()

        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            aAcc, mIoU, per_class, per_class_oa = validate(model, val_ldr, device, use_dsm)
        else:
            mIoU = hist['miou'][-1] if hist['miou'] else 0.0; per_class = []; per_class_oa = []

        hist['loss'].append(avg_loss); hist['lr'].append(opt.param_groups[0]['lr'])
        hist['miou'].append(mIoU)
        eta = time.time() - t0

        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% (best={best_v:.1f}) [{eta:.0f}s]")
            print(f"    per-class IoU: {[f'{x:.1f}' for x in per_class]}")
            print(f"    per-class OA : {[f'{x:.1f}' for x in per_class_oa]}")

        if do_val and mIoU > best_v:
            best_v = mIoU
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'best_v': best_v,
                        'hist': hist, 'args': vars(args)},
                       os.path.join(out_dir, 'best_model.pt'))
            print(f"    ✓ Saved (mIoU={best_v:.1f}%)")
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'best_v': best_v,
                        'hist': hist, 'args': vars(args)},
                       os.path.join(out_dir, f'checkpoint_epoch{epoch:03d}.pt'))

    json.dump(hist, open(os.path.join(out_dir, 'history.json'), 'w'))
    del sam3, model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Best mIoU={best_v:.1f}%\nOutput: {out_dir}")


if __name__ == '__main__':
    main()
