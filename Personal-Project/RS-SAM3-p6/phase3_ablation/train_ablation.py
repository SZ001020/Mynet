#!/usr/bin/env python3
"""Unified trainer for Plan6 Phase 3 ablation experiments.

Models: a0 | a1 | a3
Protocol: structure_loss + online crops + 20 epochs (matches Phase 1 style)
"""
from __future__ import annotations
import argparse, json, os, sys, time, numpy as np, torch, torch.nn.functional as F
from datetime import datetime

BASE = "/root/Mynet"; SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
ABLATION_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation"
sys.path.insert(0, SE)
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter")
sys.path.insert(0, ABLATION_DIR)
sys.path.insert(0, f"{ABLATION_DIR}/a0_anchor")
sys.path.insert(0, f"{ABLATION_DIR}/chain1_adapter")

from model_a1 import DSMLateFusion
from model_a3 import A3MMAdapterFull
from a0_anchor.model import A0Anchor
from structure_loss import structure_loss

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = 5
IGNORE_INDEX = 255


def load_sam3(device="cuda"):
    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device=device)
    os.chdir(prev); return m.cuda()


def build_model(model_type, sam3):
    if model_type == "a0":
        return A0Anchor(sam3, num_classes=NUM_CLASSES)
    elif model_type == "a1":
        return DSMLateFusion(sam3, num_classes=NUM_CLASSES, dsm_dim=128)
    elif model_type == "a3":
        return A3MMAdapterFull(sam3, num_classes=NUM_CLASSES, bottleneck=32, dsm_dim=128)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def make_train_dataset(epoch_steps=1000, batch_size=4):
    from dataset_adapter import VAIHINGEN_TRAIN
    tiles = VAIHINGEN_TRAIN
    img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
    gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
    dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
    dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif' for t in tiles}
    from dataset_online import OnlineCropDataset
    return OnlineCropDataset(img_dir, gt_dir, tiles, ".tif", ".tif", dsm_paths,
                             is_train=True, crop_size=256, epoch_steps=epoch_steps, batch_size=batch_size)


def make_val_dataset():
    from dataset_adapter import VAIHINGEN_VAL, _rgb_to_class
    from PIL import Image
    tiles = VAIHINGEN_VAL
    img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
    gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
    dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
    dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif' for t in tiles}
    class ValDS(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = []
            for tile in tiles:
                ip, gp, dp = f"{img_dir}/{tile}.tif", f"{gt_dir}/{tile}.tif", dsm_paths[tile]
                if not os.path.exists(ip): continue
                img = np.array(Image.open(ip).convert("RGB"))
                gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
                dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2], dtype=np.float32)
                if os.path.exists(dp): dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
                h, w = img.shape[:2]
                for y in range(0, h - 128, 128):
                    for x in range(0, w - 128, 128):
                        y2, x2 = min(y+256,h), min(x+256,w)
                        ph, pw = y2-y, x2-x
                        if ph < 128 or pw < 128: continue
                        p, l, d = img[y:y2,x:x2], gt[y:y2,x:x2], dsm[y:y2,x:x2]
                        if ph < 256 or pw < 256:
                            p = np.pad(p, ((0,256-ph),(0,256-pw),(0,0)), mode="reflect")
                            l = np.pad(l, ((0,256-ph),(0,256-pw)), mode="constant", constant_values=IGNORE_INDEX)
                            d = np.pad(d, ((0,256-ph),(0,256-pw)), mode="reflect")
                        if (l == IGNORE_INDEX).mean() <= 0.5: self.samples.append((p,d,l))
            print(f"  {len(self.samples)} fixed val windows")
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            img, dsm, label = self.samples[idx]
            return (torch.from_numpy(img.copy()).permute(2,0,1).float()/255.0,
                    torch.from_numpy(dsm.copy()).float(), torch.from_numpy(label.copy()).long())
    return ValDS()


@torch.no_grad()
def validate(model, loader, device, use_dsm):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = 0; total = 0
    for images, dsm, labels in loader:
        images, labels = images.to(device), labels.to(device)
        dsm = dsm.to(device) if use_dsm else None
        logits = model(images, dsm) if use_dsm else model(images)
        logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1); mask = labels != IGNORE_INDEX
        correct += (pred[mask] == labels[mask]).sum().item(); total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c
            inter[c] += (pc & lc).sum(); union[c] += (pc | lc).sum()
    per_class_iou = {CLASS_NAMES[c]: (inter[c] / union[c].clamp(min=1) * 100).item() for c in range(NUM_CLASSES)}
    return {"avg_oa": correct / max(total, 1) * 100, "avg_miou": float(np.mean(list(per_class_iou.values()))),
            "per_class_iou": per_class_iou}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["a0", "a1", "a3"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    device = torch.device("cuda")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan6_phase3_{args.model}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    use_dsm = args.model in ("a1", "a3")
    print(f"Plan6 Phase3 Ablation: {args.model} (use_dsm={use_dsm}, epochs={args.epochs})")
    print(f"  Output: {out_dir}")

    print("\nLoading data...")
    train_ds = make_train_dataset(epoch_steps=args.epoch_steps, batch_size=args.batch)
    val_ds = make_val_dataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nBuilding model...")
    sam3 = load_sam3()
    model = build_model(args.model, sam3).cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = {"loss": [], "metrics": [], "lr": []}
    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); start = time.time(); running_loss = 0.0
        for step, batch in enumerate(train_loader):
            images, dsm, labels = batch if len(batch) == 3 else (batch[0], None, batch[1])
            images, labels = images.to(device), labels.to(device)
            dsm = dsm.to(device) if use_dsm and dsm is not None else None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, dsm) if use_dsm and dsm is not None else model(images)
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = structure_loss(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        scheduler.step(); avg_loss = running_loss / max(len(train_loader), 1)
        metrics = validate(model, val_loader, device, use_dsm)
        print(f"  E{epoch:03d}: loss={avg_loss:.4f} OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% best={best_miou:.2f}% time={time.time()-start:.0f}s")
        print(f"    per-class IoU: {metrics['per_class_iou']}")
        if metrics["avg_miou"] > best_miou:
            best_miou = metrics["avg_miou"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_v": best_miou, "metrics": metrics, "args": vars(args)}, os.path.join(out_dir, "best_model.pt"))
        history["loss"].append(avg_loss); history["lr"].append(scheduler.get_last_lr()); history["metrics"].append(metrics)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print(f"\nDone. Best mIoU={best_miou:.2f}%  Output: {out_dir}")


if __name__ == "__main__":
    main()
