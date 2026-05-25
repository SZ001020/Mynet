#!/usr/bin/env python3
"""Evaluate Plan7-C3 with the 256x256 MFNet sliding-window protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
PHASE_C = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_c_residual_boundary/c3_full"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_A)
sys.path.insert(0, PHASE_C)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model_c3 import Plan7C3MFNet  # noqa: E402
from train_a import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
IGNORE_INDEX = 255


def load_model(checkpoint: str, adapter_bottleneck: int = 32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan7C3MFNet(
        load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=len(CLASS_NAMES),
        dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch(model, rgb: np.ndarray, dsm: np.ndarray) -> np.ndarray:
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        logits = F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def sliding_predict(model, rgb: np.ndarray, dsm: np.ndarray, window: int = 256, stride: int = 128) -> np.ndarray:
    h, w = rgb.shape[:2]
    pred_sum = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for y in range(0, h - stride, stride):
        for x in range(0, w - stride, stride):
            y2, x2 = min(y + window, h), min(x + window, w)
            ph, pw = y2 - y, x2 - x
            if ph < stride or pw < stride:
                continue
            rgb_patch = rgb[y:y2, x:x2]
            dsm_patch = dsm[y:y2, x:x2]
            if ph < window or pw < window:
                rgb_patch = np.pad(rgb_patch, ((0, window - ph), (0, window - pw), (0, 0)), mode="reflect")
                dsm_patch = np.pad(dsm_patch, ((0, window - ph), (0, window - pw)), mode="reflect")
            pred = predict_patch(model, rgb_patch, dsm_patch)[:ph, :pw]
            im, jm = min(16, ph // 4), min(16, pw // 4)
            pred_sum[y + im : y2 - im, x + jm : x2 - jm] += pred[im : ph - im, jm : pw - jm]
            count[y + im : y2 - im, x + jm : x2 - jm] += 1.0
    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path):
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)
    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) inferring...", flush=True)
    pred = sliding_predict(model, rgb, dsm)
    mask = gt != 255
    total = int(mask.sum())
    correct = int((pred[mask] == gt[mask]).sum())
    oa = float(correct / max(total, 1) * 100)
    ious, recalls = {}, {}
    counts = {}
    for idx, name in enumerate(CLASS_NAMES):
        pc, lc = pred == idx, gt == idx
        inter = float((pc & lc).sum())
        union = float((pc | lc).sum())
        gt_pixels = float(lc[mask].sum())
        ious[name] = inter / union * 100 if union > 0 else 0.0
        recalls[name] = inter / max(gt_pixels, 1.0) * 100
        counts[f"{name}_inter"] = inter
        counts[f"{name}_union"] = union
        counts[f"{name}_gt"] = gt_pixels
    miou = float(np.mean(list(ious.values())))
    mrecall = float(np.mean(list(recalls.values())))
    print(f"    OA={oa:.2f}% mIoU={miou:.2f}% mRecall={mrecall:.2f}%")
    return {
        "tile": tile, "oa": oa, "miou": miou, "mrecall": mrecall,
        "per_class_iou": ious, "per_class_recall": recalls, "counts": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan7_c3_eval_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plan7-C3 256² evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")

    model, ckpt = load_model(args.checkpoint)
    print(f"  Model epoch: {ckpt.get('epoch', 'N/A')}, best_v: {ckpt.get('best_v', 'N/A'):.2f}")

    _, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)

    all_inter = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    all_union = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    all_gt = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    all_correct = 0.0
    all_total = 0.0
    results = []

    for tile in val_t:
        dsm_path = dsm_paths.get(tile, "")
        r = evaluate_tile(model, tile, img_dir, gt_dir, img_suf, gt_suf, dsm_path)
        results.append(r)
        all_correct += r["counts"].get("correct", r["oa"] * r.get("total", 1) / 100)
        for idx in range(len(CLASS_NAMES)):
            all_inter[idx] += r["counts"].get(f"{CLASS_NAMES[idx]}_inter", 0)
            all_union[idx] += r["counts"].get(f"{CLASS_NAMES[idx]}_union", 0)
            all_gt[idx] += r["counts"].get(f"{CLASS_NAMES[idx]}_gt", 0)

    # Accumulated metrics
    per_class_iou_acc = {}
    per_class_recall_acc = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_iou_acc[name] = all_inter[idx] / max(all_union[idx], 1) * 100
        per_class_recall_acc[name] = all_inter[idx] / max(all_gt[idx], 1) * 100

    # Overall OA (per-pixel correct/total across all tiles)
    total_correct = sum(r["counts"].get("correct", 0) for r in results) if "correct" in results[0].get("counts", {}) else 0
    total_pixels = sum(r["counts"].get("total", 0) for r in results) if "total" in results[0].get("counts", {}) else 0

    # Recalculate from per-tile results
    oa_avg = np.mean([r["oa"] for r in results])
    miou_avg = np.mean([r["miou"] for r in results])
    miou_acc = float(np.mean(list(per_class_iou_acc.values())))
    mrecall_acc = float(np.mean(list(per_class_recall_acc.values())))

    summary = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_v": ckpt.get("best_v"),
        "avg_oa": oa_avg,
        "avg_miou": miou_avg,
        "acc_miou": miou_acc,
        "acc_mrecall": mrecall_acc,
        "per_class_iou": per_class_iou_acc,
        "per_class_recall": per_class_recall_acc,
        "tiles": results,
    }

    print(f"\n  === 256² Accumulated ===")
    print(f"  mIoU={miou_acc:.2f}% mRecall={mrecall_acc:.2f}% OA(tile-avg)={oa_avg:.2f}%")
    print(f"  Per-class IoU:  {per_class_iou_acc}")
    print(f"  Per-class Recall: {per_class_recall_acc}")

    json.dump(summary, open(os.path.join(out_dir, "eval_256.json"), "w"), indent=2)
    print(f"\nSaved: {out_dir}/eval_256.json")


if __name__ == "__main__":
    main()
