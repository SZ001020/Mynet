#!/usr/bin/env python3
"""Evaluate Plan6 Phase 1 with the MFNet 256x256 sliding-window protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1.5_mm_adapter"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model_phase1 import Plan6MMAdapterMFNet  # noqa: E402
from train_phase1_5 import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def load_model(checkpoint: str, adapter_bottleneck: int) -> tuple[Plan6MMAdapterMFNet, dict]:
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    adapter_bottleneck = int(ckpt_args.get("adapter_bottleneck", adapter_bottleneck))
    dsm_attn_mode = ckpt_args.get("dsm_attn_mode", "adapter")
    checkpoint_attn = bool(ckpt_args.get("checkpoint_attn", False))
    lora_rank = int(ckpt_args.get("lora_rank", ckpt_args.get("lora-rank", 0)))
    lora_alpha = float(ckpt_args.get("lora_alpha", ckpt_args.get("lora-alpha", 16.0)))

    sam3 = load_sam3()
    model = Plan6MMAdapterMFNet(
        sam3,
        adapter_bottleneck=adapter_bottleneck,
        num_classes=len(CLASS_NAMES),
        dropout=0.1,
        dsm_attn_mode=dsm_attn_mode,
        checkpoint_attn=checkpoint_attn,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch(model: Plan6MMAdapterMFNet, rgb: np.ndarray, dsm: np.ndarray) -> np.ndarray:
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        logits = F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def sliding_predict(
    model: Plan6MMAdapterMFNet,
    rgb: np.ndarray,
    dsm: np.ndarray,
    window: int = 256,
    stride: int = 128,
    crop_margin: int = 16,
) -> np.ndarray:
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
                rgb_patch = np.pad(
                    rgb_patch,
                    ((0, window - ph), (0, window - pw), (0, 0)),
                    mode="reflect",
                )
                dsm_patch = np.pad(
                    dsm_patch,
                    ((0, window - ph), (0, window - pw)),
                    mode="reflect",
                )

            pred = predict_patch(model, rgb_patch, dsm_patch)[:ph, :pw]
            im, jm = min(crop_margin, ph // 4), min(crop_margin, pw // 4)
            pred_sum[y + im : y2 - im, x + jm : x2 - jm] += pred[im : ph - im, jm : pw - jm]
            count[y + im : y2 - im, x + jm : x2 - jm] += 1.0

    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile: str, img_dir: str, gt_dir: str, img_suffix: str, gt_suffix: str, dsm_path: str):
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
    ious = {}
    recalls = {}
    counts = {}
    for idx, name in enumerate(CLASS_NAMES):
        pc = pred == idx
        lc = gt == idx
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
        "tile": tile,
        "oa": oa,
        "miou": miou,
        "correct": correct,
        "total": total,
        **ious,
        **{f"{k}_recall": v for k, v in recalls.items()},
        **counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL

    print(f"Plan6 Phase 1 MFNet protocol eval: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, args.adapter_bottleneck)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    results = [
        evaluate_tile(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths[tile])
        for tile in tiles
    ]

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = float(total_correct / max(total_pixels, 1) * 100)
    per_class_iou = {}
    per_class_recall = {}
    for name in CLASS_NAMES:
        inter = sum(r[f"{name}_inter"] for r in results)
        union = sum(r[f"{name}_union"] for r in results)
        gt_pixels = sum(r[f"{name}_gt"] for r in results)
        per_class_iou[name] = float(inter / union * 100) if union > 0 else 0.0
        per_class_recall[name] = float(inter / max(gt_pixels, 1.0) * 100)
    avg_miou = float(np.mean(list(per_class_iou.values())))
    mrecall = float(np.mean(list(per_class_recall.values())))

    summary = {
        "avg_oa": avg_oa,
        "avg_miou": avg_miou,
        "per_class_iou": per_class_iou,
        "per_class_recall": per_class_recall,
        "tiles": results,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_crop_best": ckpt.get("best_v"),
    }

    print("\nSUMMARY")
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%  mRecall={mrecall:.2f}%")
    print("  Per-class IoU: " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))
    print("  Per-class Rec: " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items()))

    out_path = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_256_{args.dataset}_plan6_phase1.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
