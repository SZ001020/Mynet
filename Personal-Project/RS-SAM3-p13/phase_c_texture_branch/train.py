#!/usr/bin/env python3
"""P13-C: RGB texture branch + vegetation refinement head.

Continues from P11-B checkpoint (76.72%), adds texture CNN + refinement head.
Tests whether raw RGB texture features help tree/grass discrimination.
"""

from __future__ import annotations

import argparse, json, os, random, sys, time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_opening

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P7_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
P11B_DIR = f"{BASE}/Personal-Project/RS-SAM3-p11/phase_b_ndsm"
P13C_DIR = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_c_texture_branch"
sys.path.insert(0, SE)
sys.path.insert(0, P7_DIR)
sys.path.insert(0, P11B_DIR)
sys.path.insert(0, P13C_DIR)

from dataset_adapter import (  # noqa: E402
    IGNORE_INDEX, NUM_CLASSES,
    POTSDAM_TRAIN, POTSDAM_VAL, VAIHINGEN_TRAIN, VAIHINGEN_VAL, _rgb_to_class,
)
from dataset_online import OnlineCropDataset, compute_ndsm  # noqa: E402
from model import Plan13CTextureMFNet  # noqa: E402
from structure_loss import structure_loss  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


class Window256DatasetDSM(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix, dsm_paths, stride=128):
        self.samples = []
        for tile in tiles:
            ip = f"{img_dir}/{tile}{img_suffix}"
            gp = f"{gt_dir}/{tile}{gt_suffix}"
            dp = dsm_paths.get(tile)
            if not os.path.exists(ip) or not os.path.exists(gp):
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            if dp and os.path.exists(dp):
                dsm_raw = np.array(Image.open(dp)).astype(np.float32)
                dsm = compute_ndsm(dsm_raw)
            else:
                dsm = np.zeros(img.shape[:2], dtype=np.float32)
            h, w = img.shape[:2]
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y + 256, h), min(x + 256, w)
                    ph, pw = y2 - y, x2 - x
                    if ph < 128 or pw < 128:
                        continue
                    patch = img[y:y2, x:x2]
                    label = gt[y:y2, x:x2]
                    dsm_patch = dsm[y:y2, x:x2]
                    if ph < 256 or pw < 256:
                        patch = np.pad(patch, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                        label = np.pad(label, ((0, 256 - ph), (0, 256 - pw)), mode="constant", constant_values=IGNORE_INDEX)
                        dsm_patch = np.pad(dsm_patch, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                    if (label == IGNORE_INDEX).mean() <= 0.5:
                        self.samples.append((patch, dsm_patch, label))
        print(f"  {len(self.samples)} fixed RGB/DSM windows (val)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, dsm, label = self.samples[idx]
        return (
            torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(dsm.copy()).float(),
            torch.from_numpy(label.copy()).long(),
        )


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


def veg_refine_loss(refined_logits, labels, weight=1.0):
    """CE loss on tree/grass within GT vegetation regions."""
    veg_gt = (labels == 2) | (labels == 3)
    if veg_gt.sum() < 1:
        return torch.tensor(0.0, device=labels.device)
    veg_logits = refined_logits[:, 2:4]  # grass=ch2, tree=ch3
    veg_logits = F.interpolate(veg_logits, labels.shape[-2:], mode="bilinear", align_corners=False)
    veg_logits_2d = veg_logits.permute(0, 2, 3, 1)[veg_gt]
    veg_labels_1d = labels[veg_gt] - 2  # grass(2)→0, tree(3)→1
    return weight * F.cross_entropy(veg_logits_2d, veg_labels_1d)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = 0; total = 0
    for images, dsm, labels in loader:
        images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
        refined, _veg_delta = model(images, dsm)
        logits = F.interpolate(refined, labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        mask = labels != IGNORE_INDEX
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c
            inter[c] += (pc & lc).sum()
            union[c] += (pc | lc).sum()
    per_class_iou = {CLASS_NAMES[c]: (inter[c] / union[c].clamp(min=1) * 100).item() for c in range(NUM_CLASSES)}
    per_class_oa = {CLASS_NAMES[c]: ((inter[c] + total - union[c]) / max(total, 1) * 100).item() for c in range(NUM_CLASSES)}
    return {
        "avg_oa": correct / max(total, 1) * 100,
        "avg_miou": float(np.mean(list(per_class_iou.values()))),
        "per_class_iou": per_class_iou, "per_class_oa": per_class_oa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--init-from", default="/root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_20260604_225146/best_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=5e-5, help="LR for texture stem + refinement head")
    parser.add_argument("--veg-weight", type=float, default=0.5, help="Weight for vegetation refinement loss")
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--dsm-attn-mode", default="full", choices=["adapter", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan13_c_texture_branch_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2, default=str)

    print("P13-C: RGB texture branch + vegetation refinement head")
    print(f"  Init from: {args.init_from}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {out_dir}")

    train_t, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)
    train_ds = OnlineCropDataset(
        img_dir, gt_dir, train_t, img_suf, gt_suf, dsm_paths,
        is_train=True, crop_size=256, epoch_steps=args.epoch_steps, batch_size=args.batch,
    )
    val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, img_suf, gt_suf, dsm_paths)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nBuilding model...")
    sam3 = load_sam3()
    model = Plan13CTextureMFNet(
        sam3,
        adapter_bottleneck=args.adapter_bottleneck,
        num_classes=NUM_CLASSES, dropout=0.1,
        dsm_attn_mode=args.dsm_attn_mode,
        checkpoint_attn=False,
        resolution=args.resolution,
    ).cuda()

    # Load P11-B base weights
    if os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location="cuda", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Loaded P11-B checkpoint (epoch {ckpt.get('epoch')}, mIoU={ckpt.get('best_v', 0):.2f}%)")
        print(f"  Missing keys: {len(missing)} (texture stem + refinement head)")
        print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"  WARNING: init-from not found, training from scratch")

    model.train()

    # Separate param groups: existing (low lr) vs new (higher lr)
    new_param_names = {"texture_stem", "veg_refine_head"}
    existing_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(n in name for n in new_param_names):
            new_params.append(param)
        else:
            existing_params.append(param)
    print(f"  Existing params: {sum(p.numel() for p in existing_params):,} (lr={args.lr})")
    print(f"  New params: {sum(p.numel() for p in new_params):,} (lr={args.new_lr})")

    optimizer = torch.optim.AdamW(
        [
            {"params": existing_params, "lr": args.lr},
            {"params": new_params, "lr": args.new_lr},
        ],
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = {"loss": [], "loss_main": [], "loss_veg": [], "metrics": [], "lr": []}
    best_miou = 0.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        running_main = 0.0
        running_veg = 0.0
        for step, (images, dsm, labels) in enumerate(train_loader):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                refined, _veg_delta = model(images, dsm)
                refined = F.interpolate(refined, labels.shape[-2:], mode="bilinear", align_corners=False)
                loss_main = structure_loss(refined.float(), labels)
                loss_veg = veg_refine_loss(refined.float(), labels, weight=args.veg_weight)
                loss = loss_main + loss_veg
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            running_main += loss_main.item()
            running_veg += loss_veg.item()
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} "
                      f"loss={loss.item():.4f} (main={loss_main.item():.4f} veg={loss_veg.item():.4f}) "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        avg_main = running_main / max(len(train_loader), 1)
        avg_veg = running_veg / max(len(train_loader), 1)
        metrics = None
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate(model, val_loader, device)
            print(f"  E{epoch:03d}: loss={avg_loss:.4f} (main={avg_main:.4f} veg={avg_veg:.4f}) "
                  f"OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% "
                  f"best={best_miou:.2f}% time={time.time() - start:.0f}s")
            print(f"    per-class IoU: {metrics['per_class_iou']}")
            if metrics["avg_miou"] > best_miou:
                best_miou = metrics["avg_miou"]
                best_metrics = metrics
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "best_v": best_miou, "metrics": metrics, "args": vars(args)},
                    os.path.join(out_dir, "best_model.pt"),
                )
        history["loss"].append(avg_loss)
        history["loss_main"].append(avg_main)
        history["loss_veg"].append(avg_veg)
        history["lr"].append(scheduler.get_last_lr())
        history["metrics"].append(metrics)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
        if best_metrics is not None:
            json.dump(best_metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    print(f"\nDone. Best mIoU={best_miou:.2f}%")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
