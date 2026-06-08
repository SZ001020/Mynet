#!/usr/bin/env python3
"""Evaluate Plan7-A with soft-logit sliding window + 5×5 confusion matrix."""

from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model import Plan7PromptMFNet  # noqa: E402
from train_a import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint: str, adapter_bottleneck: int = 32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan7PromptMFNet(
        load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=NUM_CLASSES,
        dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def pad_to_size(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= target_h and w >= target_w:
        return img
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if img.ndim == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")


@torch.no_grad()
def predict_patch_logits(model, rgb_patch: np.ndarray, dsm_patch: np.ndarray) -> np.ndarray:
    rgb_t = torch.from_numpy(rgb_patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm_patch).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().numpy()


@torch.no_grad()
def sliding_soft_logits(
    model, img_np: np.ndarray, dsm_np: np.ndarray,
    window: int = 256, stride: int = 128,
) -> np.ndarray:
    """Accumulate soft logits over sliding windows, return logits [C, H, W]."""
    H, W = img_np.shape[:2]
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - stride, stride):
        for x in range(0, W - stride, stride):
            y2 = min(y + window, H)
            x2 = min(x + window, W)
            ph, pw = y2 - y, x2 - x
            if ph < stride or pw < stride:
                continue

            patch_rgb = img_np[y:y2, x:x2]
            patch_dsm = dsm_np[y:y2, x:x2]
            if ph < window or pw < window:
                patch_rgb = pad_to_size(patch_rgb, window, window)
                patch_dsm = pad_to_size(patch_dsm, window, window)

            logits = predict_patch_logits(model, patch_rgb, patch_dsm)[:, :ph, :pw]
            trim = min(16, ph // 4, pw // 4)
            yy, xx = y + trim, x + trim
            yy2, xx2 = y2 - trim, x2 - trim
            if yy2 > yy and xx2 > xx:
                logit_sum[:, yy:yy2, xx:xx2] += logits[:, trim:ph - trim, trim:pw - trim]
                count[yy:yy2, xx:xx2] += 1.0

    count[count == 0] = 1.0
    logit_sum /= count
    return logit_sum


def evaluate_tile_with_cm(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path):
    """Return tile result dict including a 5×5 confusion matrix (raw counts)."""
    t0 = time.time()
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt_full = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)

    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) inferring...", end=" ", flush=True)
    logits = sliding_soft_logits(model, rgb, dsm)
    pred = logits.argmax(0).astype(np.int64)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")

    mask = gt_full != 255
    gt = gt_full[mask]
    pd = pred[mask]
    total = int(mask.sum())
    correct = int((pd == gt).sum())
    oa = float(correct / max(total, 1) * 100)

    # Build 5×5 confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        gt_i = (gt == i)
        for j in range(NUM_CLASSES):
            cm[i, j] = int((gt_i & (pd == j)).sum())

    # Per-class metrics from confusion matrix
    ious, recalls, precisions, f1s = {}, {}, {}, {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        union = tp + fp + fn
        ious[name] = float(tp / union * 100) if union > 0 else 0.0
        recalls[name] = float(tp / max(tp + fn, 1) * 100)
        precisions[name] = float(tp / max(tp + fp, 1) * 100)
        f1s[name] = float(2 * precisions[name] * recalls[name] / max(precisions[name] + recalls[name], 1e-8))

    miou = float(np.mean(list(ious.values())))

    return {
        "tile": tile, "oa": oa, "miou": miou,
        "correct": correct, "total": total,
        "per_class_iou": ious,
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "confusion_matrix": cm.tolist(),
    }


def print_confusion_matrix(cm: np.ndarray, title: str, row_labels: list, col_labels: list):
    """Pretty-print a confusion matrix."""
    print(f"\n{title}")
    header = "GT \\ Pred  " + "".join(f"{c:>10s}" for c in col_labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(row_labels):
        row = f"{label:10s}  " + "".join(f"{cm[i,j]:10d}" for j in range(len(col_labels)))
        print(row)


def print_normalized_cm(cm: np.ndarray, title: str, labels: list, mode: str = "recall"):
    """Print row-normalized (recall) or column-normalized (precision) confusion matrix."""
    if mode == "recall":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = cm / row_sums * 100
    else:
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        mat = cm / col_sums * 100

    print(f"\n{title}")
    header = "GT \\ Pred  " + "".join(f"{c:>8s}" for c in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:10s}  " + "".join(f"{mat[i,j]:8.1f}" for j in range(len(labels)))
        print(row)


def analyze_confusion(cm: np.ndarray):
    """Print diagnostic analysis of the confusion matrix."""
    print("\n" + "=" * 70)
    print("CONFUSION DIAGNOSIS")
    print("=" * 70)

    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        total_gt = cm[i, :].sum()
        fn_indices = np.argsort(cm[i, :])[::-1]
        fn_indices = [idx for idx in fn_indices if idx != i and cm[i, idx] > 0][:3]

        fp_indices = np.argsort(cm[:, i])[::-1]
        fp_indices = [idx for idx in fp_indices if idx != i and cm[idx, i] > 0][:3]

        recall = tp / max(total_gt, 1) * 100
        fn_rate = 100 - recall

        print(f"\n  [{name}] recall={recall:.1f}%  (misses {fn_rate:.1f}% of {name} pixels)")
        if fn_indices:
            fn_str = ", ".join(f"{CLASS_NAMES[j]}({cm[i,j]/max(total_gt,1)*100:.1f}%)" for j in fn_indices)
            print(f"    → misclassified as: {fn_str}")
        if fp_indices:
            total_pred = cm[:, i].sum()
            fp_str = ", ".join(f"{CLASS_NAMES[j]}({cm[j,i]/max(total_pred,1)*100:.1f}% of pred)" for j in fp_indices)
            print(f"    ← false positives from: {fp_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL

    print(f"Plan7-A confusion matrix eval: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, args.adapter_bottleneck)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    results = [
        evaluate_tile_with_cm(model, t, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths[t])
        for t in tiles
    ]

    # Aggregate
    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = float(total_correct / max(total_pixels, 1) * 100)

    # Global confusion matrix
    global_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for r in results:
        global_cm += np.array(r["confusion_matrix"], dtype=np.int64)

    # Global per-class metrics
    per_class_iou, per_class_recall, per_class_precision, per_class_f1 = {}, {}, {}, {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(global_cm[i, i])
        fp = int(global_cm[:, i].sum() - tp)
        fn = int(global_cm[i, :].sum() - tp)
        union = tp + fp + fn
        per_class_iou[name] = float(tp / union * 100) if union > 0 else 0.0
        per_class_recall[name] = float(tp / max(tp + fn, 1) * 100)
        per_class_precision[name] = float(tp / max(tp + fp, 1) * 100)
        per_class_f1[name] = float(2 * per_class_precision[name] * per_class_recall[name] /
                                   max(per_class_precision[name] + per_class_recall[name], 1e-8))

    avg_miou = float(np.mean(list(per_class_iou.values())))

    # Print everything
    print_confusion_matrix(global_cm, "GLOBAL CONFUSION MATRIX (raw counts)", CLASS_NAMES, CLASS_NAMES)
    print_normalized_cm(global_cm, "ROW-NORMALIZED (% of GT pixels → Pred, i.e. Recall view)", CLASS_NAMES, "recall")
    print_normalized_cm(global_cm, "COLUMN-NORMALIZED (% of Pred pixels ← GT, i.e. Precision view)", CLASS_NAMES, "precision")
    analyze_confusion(global_cm)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print(f"  {'Class':>12s}  {'IoU':>7s}  {'Recall':>7s}  {'Precision':>9s}  {'F1':>7s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}")
    for name in CLASS_NAMES:
        print(f"  {name:>12s}  {per_class_iou[name]:6.2f}%  {per_class_recall[name]:6.2f}%  "
              f"{per_class_precision[name]:8.2f}%  {per_class_f1[name]:6.2f}%")

    # Save
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_crop_best": ckpt.get("best_v"),
        "dataset": args.dataset,
        "protocol": "256² sliding window, soft-logit accumulation",
        "avg_oa": avg_oa,
        "avg_miou": avg_miou,
        "per_class_iou": per_class_iou,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": global_cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "per_tile": results,
    }
    out = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_confusion_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
