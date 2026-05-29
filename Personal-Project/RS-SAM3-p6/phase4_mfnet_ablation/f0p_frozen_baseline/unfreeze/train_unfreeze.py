#!/usr/bin/env python3
"""Train U1/U2/U3: Selective deep-layer unfreezing on SAM3 ViTDet.

U1: unfreeze blocks 24-31 attn (qkv+proj)  | attn_lr=1e-5
U2: unfreeze blocks 24-31 attn+MLP          | attn_lr=1e-5
U3: unfreeze blocks 28-31 attn              | attn_lr=1e-5 (Phase 1.6 repl, fixed windows)

All use F0' architecture (frozen SAM3 + 4xSEFusion + MFNetDecoder),
FIXED training windows (no online crops), 12 epochs, seed=42.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
F0P = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
sys.path.extend([SE, PHASE1, F0P, SHARED])

from dataset_adapter import (IGNORE_INDEX, NUM_CLASSES, VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                              POTSDAM_TRAIN, POTSDAM_VAL, _rgb_to_class)
from model_f0p import FrozenSAM3DFM
from structure_loss import structure_loss

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def dataset_paths(dataset):
    if dataset == "vaihingen":
        tiles_train, tiles_val = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suffix, gt_suffix = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area", "")}.tif'
                     for t in tiles_train + tiles_val}
    else:
        tiles_train, tiles_val = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suffix, gt_suffix = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_", "")}.tif'
                     for t in tiles_train + tiles_val}
    return tiles_train, tiles_val, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths


def load_sam3():
    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev)
    return m.cuda()


class FixedWindowDS(torch.utils.data.Dataset):
    """Fixed window dataset — NO randomness, same windows every epoch."""

    def __init__(self, img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths, stride=128):
        self.samples = []
        for tile in tiles:
            ip = f"{img_dir}/{tile}{img_suf}"
            gp = f"{gt_dir}/{tile}{gt_suf}"
            dp = dsm_paths[tile]
            if not os.path.exists(ip):
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2])
            dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
            h, w = img.shape[:2]
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y + 256, h), min(x + 256, w)
                    ph, pw = y2 - y, x2 - x
                    if ph < 128 or pw < 128:
                        continue
                    p, d, l = img[y:y2, x:x2], dsm[y:y2, x:x2], gt[y:y2, x:x2]
                    if ph < 256 or pw < 256:
                        p = np.pad(p, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                        l = np.pad(l, ((0, 256 - ph), (0, 256 - pw)), "constant", constant_values=IGNORE_INDEX)
                        d = np.pad(d, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                    if (l == IGNORE_INDEX).mean() <= 0.5:
                        self.samples.append((p, d, l))
        print(f"  {len(self.samples)} fixed windows (train)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, dsm, label = self.samples[idx]
        return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(dsm.copy()).float(),
                torch.from_numpy(label.copy()).long())


def make_val_ds(img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths, stride=128):
    samples = []
    for tile in tiles:
        ip = f"{img_dir}/{tile}{img_suf}"; gp = f"{gt_dir}/{tile}{gt_suf}"
        dp = dsm_paths[tile]
        if not os.path.exists(ip): continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
        dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2])
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        h, w = img.shape[:2]
        for y in range(0, h - 128, stride):
            for x in range(0, w - 128, stride):
                y2, x2 = min(y + 256, h), min(x + 256, w); ph, pw = y2 - y, x2 - x
                if ph < 128 or pw < 128: continue
                p, d, l = img[y:y2, x:x2], dsm[y:y2, x:x2], gt[y:y2, x:x2]
                if ph < 256 or pw < 256:
                    p = np.pad(p, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                    l = np.pad(l, ((0, 256 - ph), (0, 256 - pw)), "constant", constant_values=IGNORE_INDEX)
                    d = np.pad(d, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                if (l == IGNORE_INDEX).mean() <= 0.5: samples.append((p, d, l))
    print(f"  {len(samples)} val windows")
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            img, dsm, label = samples[i]
            return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(dsm.copy()).float(),
                    torch.from_numpy(label.copy()).long())
    return DS()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct, total = 0, 0
    for images, dsm, labels in loader:
        images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
        logits = model(images, dsm)
        logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        mask = labels != IGNORE_INDEX
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c
            inter[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()
    pci = {CLASS_NAMES[c]: (inter[c] / union[c].clamp(min=1) * 100).item() for c in range(NUM_CLASSES)}
    return {"avg_oa": correct / max(total, 1) * 100, "avg_miou": float(np.mean(list(pci.values()))),
            "per_class_iou": pci}


def unfreeze_blocks(model, start_block, end_block, unfreeze_mlp=False):
    """Set requires_grad=True on specified ViTDet blocks."""
    trunk = model.backbone.vision_backbone.trunk
    unfrozen = 0
    for idx in range(start_block, min(end_block + 1, len(trunk.blocks))):
        blk = trunk.blocks[idx]
        for n, p in blk.named_parameters():
            # Attention params
            if 'attn' in n:
                p.requires_grad = True
                unfrozen += 1
            # MLP params
            if unfreeze_mlp and 'mlp' in n:
                p.requires_grad = True
                unfrozen += 1
    print(f"  Unfroze {unfrozen} params in blocks {start_block}-{end_block}"
          + (" (attn+MLP)" if unfreeze_mlp else " (attn only)"))


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    print(f"  Loaded F0' best: missing={len(missing)}, unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--attn-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--unfreeze-start", type=int, default=24)
    parser.add_argument("--unfreeze-end", type=int, default=31)
    parser.add_argument("--unfreeze-mlp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    parser.add_argument("--init-from", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"u{args.unfreeze_start}_{args.unfreeze_end}" + ("_mlp" if args.unfreeze_mlp else "")
    out_dir = os.path.join(args.output, f"plan6_phase4_{tag}_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print(f"Plan6 Phase4 Unfreeze: blocks {args.unfreeze_start}-{args.unfreeze_end}"
          + (" (attn+MLP)" if args.unfreeze_mlp else " (attn only)"))
    print(f"  Dataset: {args.dataset} | seed={args.seed} | epochs={args.epochs}")
    print(f"  attn_lr={args.attn_lr}, decoder_lr={args.decoder_lr}")

    train_t, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)
    train_ds = FixedWindowDS(img_dir, gt_dir, train_t, img_suf, gt_suf, dsm_paths)
    val_ds = make_val_ds(img_dir, gt_dir, val_t, img_suf, gt_suf, dsm_paths)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nBuilding model...")
    sam3 = load_sam3()
    model = FrozenSAM3DFM(sam3, num_classes=NUM_CLASSES, dropout=0.1, resolution=args.resolution).cuda()
    load_checkpoint(model, args.init_from)
    unfreeze_blocks(model, args.unfreeze_start, args.unfreeze_end, args.unfreeze_mlp)
    model.train()

    # Separate params: unfrozen backbone vs decoder
    unfrozen_ids = {id(p) for p in model.backbone.parameters() if p.requires_grad}
    attn_params = [p for p in model.parameters() if p.requires_grad and id(p) in unfrozen_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in unfrozen_ids]
    print(f"  attn params: {sum(p.numel() for p in attn_params):,}, other: {sum(p.numel() for p in other_params):,}")

    opt = torch.optim.AdamW([
        {"params": attn_params, "lr": args.attn_lr},
        {"params": other_params, "lr": args.decoder_lr},
    ], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_miou = 0.0
    history = {"loss": [], "metrics": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for step, (images, dsm, labels) in enumerate(train_loader):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, dsm)
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = structure_loss(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d} loss={loss.item():.4f}")

        sch.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        metrics = validate(model, val_loader, device)
        print(f"  E{epoch:03d}: loss={avg_loss:.4f} OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% "
              f"best={best_miou:.2f}% time={time.time() - start:.0f}s")
        if metrics["avg_miou"] > best_miou:
            best_miou = metrics["avg_miou"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_v": best_miou, "metrics": metrics,
                        "args": vars(args)}, os.path.join(out_dir, "best_model.pt"))

        history["loss"].append(avg_loss)
        history["metrics"].append(metrics)
        history["lr"].append(sch.get_last_lr()[0])
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)

    print(f"\nDone. Best mIoU={best_miou:.2f}% | Run: {out_dir}")


if __name__ == "__main__":
    main()
