#!/usr/bin/env python3
"""Evaluate Plan6 Phase 1.6 with the MFNet 256x256 sliding-window protocol."""

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
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1.6_unfreeze_attn"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model_phase1 import Plan6MMAdapterMFNet  # noqa: E402
from train_phase1_6 import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def load_model(checkpoint: str) -> tuple[Plan6MMAdapterMFNet, dict]:
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    adapter_bottleneck = int(ckpt_args.get("adapter_bottleneck", 32))
    dsm_attn_mode = ckpt_args.get("dsm_attn_mode", "adapter")
    checkpoint_attn = bool(ckpt_args.get("checkpoint_attn", False))
    unfreeze_layers = int(ckpt_args.get("unfreeze_layers", 0))

    sam3 = load_sam3()
    model = Plan6MMAdapterMFNet(
        sam3,
        adapter_bottleneck=adapter_bottleneck,
        num_classes=len(CLASS_NAMES),
        dropout=0.1,
        dsm_attn_mode=dsm_attn_mode,
        checkpoint_attn=checkpoint_attn,
        unfreeze_layers=unfreeze_layers,
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch(model: Plan6MMAdapterMFNet, rgb: np.ndarray, dsm: np.ndarray) -> np.ndarray:
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(), dsm_t.cuda())
    logits = F.interpolate(logits, (256, 256), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def sliding_predict(model, img_np: np.ndarray, dsm_np: np.ndarray, window_size: int = 256, stride: int = 128):
    H, W = img_np.shape[:2]
    pred_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + window_size, H), min(x + window_size, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue
            rgb_patch = img_np[y:y2, x:x2]
            dsm_patch = dsm_np[y:y2, x:x2]
            pred = predict_patch(model, rgb_patch, dsm_patch)
            m = 16
            im, jm = min(m, ph // 4), min(m, pw // 4)
            pred_sum[y + im:y2 - im, x + jm:x2 - jm] += pred[im:ph - im, jm:pw - jm]
            count[y + im:y2 - im, x + jm:x2 - jm] += 1.0
    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile: str, img_dir: str, gt_dir: str, img_suffix: str, gt_suffix: str, dsm_path: str):
    img = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(img.shape[:2], dtype=np.float32)

    print(f"  {tile} ({img.shape[1]}x{img.shape[0]}) inferring...")
    pred = sliding_predict(model, img, dsm)
    mask = gt != 255
    total = int(mask.sum())
    correct = int((pred[mask] == gt[mask]).sum())
    oa = correct / max(total, 1) * 100
    result = {"tile": tile, "oa": float(oa), "miou": None, "n_windows": 0}
    ious = {}
    recalls = {}
    counts = {}
    for i, c in enumerate(CLASS_NAMES):
        pc, lc = pred == i, gt == i
        inter = float((pc & lc).sum())
        union = float((pc | lc).sum())
        gt_pixels = float(lc[mask].sum())
        ious[c] = float((inter / union * 100) if union > 0 else 0.0)
        recalls[c] = float(inter / max(gt_pixels, 1.0) * 100)
        counts[f"{c}_inter"] = inter
        counts[f"{c}_union"] = union
        counts[f"{c}_gt"] = gt_pixels
        result[c] = ious[c]
        result[f"{c}_recall"] = recalls[c]
        result.update(counts)
    result["miou"] = float(np.mean(list(ious.values())))
    mrecall = float(np.mean(list(recalls.values())))
    result["correct"] = correct
    result["total"] = total
    print(f"    OA={oa:.2f}% mIoU={result['miou']:.2f}% mRecall={mrecall:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL

    print(f"Plan6 Phase 1.6 MFNet protocol eval: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint)
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
        "avg_oa": avg_oa, "avg_miou": avg_miou,
        "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
        "tiles": results, "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_crop_best": ckpt.get("best_v"),
    }

    print("\nSUMMARY")
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%  mRecall={mrecall:.2f}%")
    print("  Per-class IoU: " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))
    print("  Per-class Rec: " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items()))

    out_path = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_256_{args.dataset}_plan6_phase1_6.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
