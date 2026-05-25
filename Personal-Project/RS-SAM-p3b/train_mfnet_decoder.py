#!/usr/bin/env python3
"""
VPT Adapter + MFNet full Decoder + 256² window training.

Key upgrade: Replace simplified UNetFormer decoder with MFNet's complete
DFM (4-scale SEFusion pyramid) + Decoder (GLA blocks + PA/CA attention).

用法:
  cd /root/Mynet/RS-SAM-p3b
  python train_mfnet_decoder.py --model dsm --epochs 20 --batch 4
"""

import os, sys, json, time, gc, argparse, random
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p3r')
sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM-p3b')

from structure_loss import structure_loss
from adapter_vit import inject_adapters
from mfnet_decoder import MFNetDecoder, Pyramid4Scale, SEFusion
from dataset_adapter import (_rgb_to_class, NUM_CLASSES, IGNORE_INDEX,
                             VAIHINGEN_TRAIN, VAIHINGEN_VAL)


# ── Reuse 256² Window Dataset from p3r ──────────────────────

from train_256 import Window256Dataset, Window256DatasetDSM


# ── VPT + MFNet Decoder Model ───────────────────────────────

class VPT_MFNetDecoder(nn.Module):
    """VPT Adapter on SAM3 ViT + MFNet's complete DFM + Decoder.

    RGB: SAM3 ViT (VPT) → Pyramid4Scale (4 scales)
    DSM (optional): CNN encoder → Pyramid4Scale (4 scales)
    Fusion: SEFusion at each scale
    Decoder: MFNet Decoder (GLA blocks + PA/CA)
    """

    def __init__(self, sam3_model, adapter_bottleneck=32, num_classes=5,
                 use_dsm=True, dropout=0.1, window_size=8):
        super().__init__()
        self.backbone = sam3_model.backbone
        self.use_dsm = use_dsm

        # VPT Adapters
        inject_adapters(self.backbone.vision_backbone, bottleneck=adapter_bottleneck)
        for n, p in self.backbone.vision_backbone.named_parameters():
            if 'prompt_learn' not in n:
                p.requires_grad = False
        for p in self.backbone.language_backbone.parameters():
            p.requires_grad = False

        # 4-scale pyramid for RGB
        self.pyramid_rgb = Pyramid4Scale(256)

        # DSM encoder + pyramid
        if use_dsm:
            from adapter_vit import DSMEncoderDeep
            self.dsm_encoder = DSMEncoderDeep(dsm_dim=128)
            self.pyramid_dsm = Pyramid4Scale(128)
            # SEFusion at each of 4 scales
            self.fusion1 = SEFusion(256)
            self.fusion2 = SEFusion(256)
            self.fusion3 = SEFusion(256)
            self.fusion4 = SEFusion(256)
            # DSM scale projection: 128 → 256
            self.dsm_proj = nn.ModuleList([nn.Conv2d(128, 256, 1) for _ in range(4)])

        # MFNet Decoder
        self.decoder = MFNetDecoder(num_classes=num_classes, decode_channels=64,
                                    dropout=dropout, window_size=window_size)

        self._log_params()

    def _log_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        adapter_p = sum(p.numel() for n, p in self.backbone.vision_backbone.named_parameters()
                        if 'prompt_learn' in n)
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        print(f"  VPT: {adapter_p:,} | Decoder: {decoder_p:,}")
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def extract_vit_features(self, images, resolution=1008):
        images_norm = (images - 0.5) / 0.5
        _, _, h, w = images_norm.shape
        if h != resolution or w != resolution:
            images_norm = F.interpolate(images_norm, (resolution, resolution),
                                        mode='bilinear', align_corners=False)
        out = self.backbone.forward_image(images_norm)
        return out['backbone_fpn']  # 3 FPN scales from SAM3

    def forward(self, images, dsm=None):
        # Get SAM3 ViT's deepest feature for pyramid
        fpn = self.extract_vit_features(images)
        vit_feat = fpn[-1]  # deepest: 72² (use this for 4-scale pyramid)

        # RGB 4-scale pyramid
        scales_rgb = self.pyramid_rgb(vit_feat)

        if dsm is not None and self.use_dsm:
            dsm_out = self.dsm_encoder(dsm)
            dsm_feat = dsm_out['features']
            # Align DSM spatial size to match SAM3 ViT output
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


# ── Training ────────────────────────────────────────────────

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
    return aAcc, np.mean(pc_iou), pc_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dsm', choices=['rgb', 'dsm'])
    parser.add_argument('--dataset', default='vaihingen')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--output', default='/root/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output, f'plan3_mfnetdec_{args.model}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    use_dsm = (args.model == 'dsm')
    print(f"VPT + MFNet Decoder + 256² Windows {'+DSM' if use_dsm else ''}")
    print(f"  Dataset: {args.dataset}, Epochs: {args.epochs}")
    print(f"  Output: {out_dir}")

    # Data
    print("\nLoading data...")
    train_t, val_t = VAIHINGEN_TRAIN, VAIHINGEN_VAL
    img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
    gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
    isuf, gsuf = '.tif', '.tif'

    if use_dsm:
        dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
        dsm_tiles = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif'
                     for t in train_t + val_t}
        train_ds = Window256DatasetDSM(img_dir, gt_dir, train_t, isuf, gsuf, dsm_tiles, True)
        val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, isuf, gsuf, dsm_tiles, False)
    else:
        train_ds = Window256Dataset(img_dir, gt_dir, train_t, isuf, gsuf, True)
        val_ds = Window256Dataset(img_dir, gt_dir, val_t, isuf, gsuf, False)

    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                            num_workers=0, drop_last=True, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} samples, {len(train_ldr)} batches")

    # Model
    print("\nBuilding model...")
    sam3 = load_sam3(device)
    model = VPT_MFNetDecoder(sam3, adapter_bottleneck=32, num_classes=NUM_CLASSES,
                             use_dsm=use_dsm, dropout=0.1).cuda()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_v = 1, 0.0
    hist = {'loss': [], 'lr': [], 'miou': []}

    print(f"\n{'='*60}")
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

            logits = model(images, dsm) if use_dsm else model(images)
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
                      f"loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e}")

        sch.step()
        avg_loss = ep_loss / max(n_batches, 1)
        do_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            aAcc, mIoU, per_class = validate(model, val_ldr, device, use_dsm)
        else:
            mIoU = hist['miou'][-1] if hist['miou'] else 0.0; per_class = []

        hist['loss'].append(avg_loss); hist['lr'].append(sch.get_last_lr()[0])
        hist['miou'].append(mIoU)
        eta = time.time() - t0
        if do_val:
            print(f"  E{epoch:3d}: loss={avg_loss:.4f} mIoU={mIoU:.1f}% (best={best_v:.1f}) [{eta:.0f}s]")
            print(f"    per-class: {[f'{x:.1f}' for x in per_class]}")
        if do_val and mIoU > best_v:
            best_v = mIoU
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
                        'best_v': best_v, 'hist': hist, 'args': vars(args)},
                       os.path.join(out_dir, 'best_model.pt'))
            print(f"    ✓ Saved (mIoU={best_v:.1f}%)")
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
