#!/usr/bin/env python3
"""P11-A: Train with tree/grass boundary-weighted loss on top of Plan7-A checkpoint."""

from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE7_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
P11A_DIR = f"{BASE}/Personal-Project/RS-SAM3-p11/phase_a_veg_boundary_loss"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE7_DIR)   # fallback: model, dataset, etc.
sys.path.insert(0, P11A_DIR)     # P11-A modified structure_loss takes precedence (must be FIRST)

from dataset_adapter import (  # noqa: E402
    IGNORE_INDEX, NUM_CLASSES,
    POTSDAM_TRAIN, POTSDAM_VAL, VAIHINGEN_TRAIN, VAIHINGEN_VAL, _rgb_to_class,
)
from dataset_online import OnlineCropDataset  # noqa: E402
from model import Plan7PromptMFNet  # noqa: E402
from structure_loss import structure_loss  # noqa: E402 (P11-A version)

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
                dsm = np.array(Image.open(dp)).astype(np.float32)
                dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
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


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = 0
    total = 0
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
    per_class_iou = {CLASS_NAMES[c]: (inter[c] / union[c].clamp(min=1) * 100).item() for c in range(NUM_CLASSES)}
    per_class_oa = {CLASS_NAMES[c]: ((inter[c] + total - union[c]) / max(total, 1) * 100).item() for c in range(NUM_CLASSES)}
    return {
        "avg_oa": correct / max(total, 1) * 100,
        "avg_miou": float(np.mean(list(per_class_iou.values()))),
        "per_class_iou": per_class_iou,
        "per_class_oa": per_class_oa,
    }


def load_initial_checkpoint(model, path: str):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  Initialized from: {path}")
    print(f"  Missing keys: {len(missing)} (expected for prompt branch)")
    print(f"  Unexpected keys: {len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dsm-lr", type=float, default=2.5e-5)
    parser.add_argument("--prompt-lr", type=float, default=2.5e-5)
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--dsm-attn-mode", default="full", choices=["adapter", "full"])
    parser.add_argument("--checkpoint-attn", action="store_true")
    parser.add_argument("--veg-boundary-weight", type=float, default=3.0,
                        help="Extra weight multiplier at tree-grass boundaries (default 3.0)")
    parser.add_argument(
        "--init-from",
        default="/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_225309/best_model.pt",
    )
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    device = torch.device("cuda")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan11_a_veg_boundary_loss_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print("P11-A: Tree/Grass boundary-weighted structure loss")
    print(f"  veg_boundary_weight: {args.veg_boundary_weight}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {out_dir}")
    print(f"  Online crops/epoch: {args.batch * args.epoch_steps}")

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
    model = Plan7PromptMFNet(
        sam3,
        adapter_bottleneck=args.adapter_bottleneck,
        num_classes=NUM_CLASSES, dropout=0.1,
        dsm_attn_mode=args.dsm_attn_mode,
        checkpoint_attn=args.checkpoint_attn,
        resolution=args.resolution,
    ).cuda()
    load_initial_checkpoint(model, args.init_from)
    model.train()

    prompt_params = list(model.prompt_encoder.parameters())
    prompt_ids = {id(p) for p in prompt_params}
    dsm_params = list(model.dsm_encoder.parameters())
    dsm_ids = {id(p) for p in dsm_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in prompt_ids and id(p) not in dsm_ids]
    print(
        f"  Optim params: other={sum(p.numel() for p in other_params):,}, "
        f"dsm={sum(p.numel() for p in dsm_params):,}, prompt={sum(p.numel() for p in prompt_params):,}"
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": dsm_params, "lr": args.dsm_lr},
            {"params": prompt_params, "lr": args.prompt_lr},
        ],
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = {"loss": [], "metrics": [], "lr": []}
    best_miou = 0.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for step, (images, dsm, labels) in enumerate(train_loader):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, dsm)
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = structure_loss(logits.float(), labels,
                                      veg_boundary_weight=args.veg_boundary_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        metrics = None
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate(model, val_loader, device)
            print(f"  E{epoch:03d}: loss={avg_loss:.4f} "
                  f"OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% "
                  f"best={best_miou:.2f}% time={time.time() - start:.0f}s")
            print(f"    per-class IoU: {metrics['per_class_iou']}")
            if metrics["avg_miou"] > best_miou:
                best_miou = metrics["avg_miou"]
                best_metrics = metrics
                torch.save(
                    {
                        "epoch": epoch, "model": model.state_dict(),
                        "best_v": best_miou, "metrics": metrics,
                        "args": vars(args),
                    },
                    os.path.join(out_dir, "best_model.pt"),
                )
        history["loss"].append(avg_loss)
        history["lr"].append(scheduler.get_last_lr())
        history["metrics"].append(metrics)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
        if best_metrics is not None:
            json.dump(best_metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    print(f"\nDone. Best mIoU={best_miou:.2f}%")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
