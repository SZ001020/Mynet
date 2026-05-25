#!/usr/bin/env python3
"""Train F0: MFNet exact architecture on SAM3 (shared encoder + LoRA + SEFusion + MFNetDecoder)."""

from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE1_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
F0_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE1_DIR)
sys.path.insert(0, F0_DIR)
sys.path.insert(0, SHARED)

from dataset_adapter import (  # noqa: E402
    IGNORE_INDEX, NUM_CLASSES, VAIHINGEN_TRAIN, VAIHINGEN_VAL, POTSDAM_TRAIN, POTSDAM_VAL,
    _rgb_to_class,
)
from model_f0 import MFNetSAM3  # noqa: E402
from structure_loss import structure_loss  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def dataset_paths(dataset: str):
    if dataset == "vaihingen":
        tiles_train, tiles_val = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suffix, gt_suffix = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {
            t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area", "")}.tif'
            for t in tiles_train + tiles_val
        }
    else:
        tiles_train, tiles_val = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suffix, gt_suffix = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {
            t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_", "")}.tif'
            for t in tiles_train + tiles_val
        }
    return tiles_train, tiles_val, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths


def load_sam3(device: str = "cuda"):
    prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt",
        device=device,
    )
    os.chdir(prev)
    return model.cuda()


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


class OnlineCropDataset(torch.utils.data.Dataset):
    """Online random crops from training tiles."""

    def __init__(self, img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths,
                 crop_size=256, epoch_steps=1000, batch_size=2):
        self.crop_size = crop_size
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.tiles = tiles
        self.total = epoch_steps * batch_size

        self.cache = []
        for tile in tiles:
            ip = f"{img_dir}/{tile}{img_suf}"
            gp = f"{gt_dir}/{tile}{gt_suf}"
            dp = dsm_paths[tile]
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            dsm = np.array(Image.open(dp)).astype(np.float32)
            dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
            self.cache.append((img, gt, dsm))
        print(f"  Cached {len(tiles)} tiles for train; online crops/epoch={self.total}")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img, gt, dsm = self.cache[np.random.randint(0, len(self.cache))]
        h, w = img.shape[:2]
        cs = self.crop_size
        if h > cs and w > cs:
            y = np.random.randint(0, h - cs)
            x = np.random.randint(0, w - cs)
            patch = img[y:y + cs, x:x + cs]
            label = gt[y:y + cs, x:x + cs]
            dsm_p = dsm[y:y + cs, x:x + cs]
        else:
            patch, label, dsm_p = img, gt, dsm
        return (torch.from_numpy(patch.copy()).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(dsm_p.copy()).float(),
                torch.from_numpy(label.copy()).long())


def make_val_ds(img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths, stride=128):
    samples = []
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
                    samples.append((p, d, l))
    print(f"  {len(samples)} val windows")

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            img, dsm, label = samples[i]
            return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(dsm.copy()).float(),
                    torch.from_numpy(label.copy()).long())

    return DS()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan6_phase4_f0_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print("Plan6 Phase4 F0: MFNet on SAM3 (shared encoder + LoRA + SEFusion + MFNetDecoder)")
    print(f"  Dataset: {args.dataset} | seed={args.seed} | epochs={args.epochs}")

    train_t, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)
    train_ds = OnlineCropDataset(img_dir, gt_dir, train_t, img_suf, gt_suf, dsm_paths,
                                 epoch_steps=args.epoch_steps, batch_size=args.batch)
    val_ds = make_val_ds(img_dir, gt_dir, val_t, img_suf, gt_suf, dsm_paths)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                                               num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nBuilding model...")
    sam3 = load_sam3()
    model = MFNetSAM3(sam3, lora_rank=args.lora_rank, num_classes=NUM_CLASSES,
                      dropout=0.1, resolution=args.resolution).cuda()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = {"loss": [], "metrics": [], "lr": []}
    best_miou = 0.0

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
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} loss={loss.item():.4f}")

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
