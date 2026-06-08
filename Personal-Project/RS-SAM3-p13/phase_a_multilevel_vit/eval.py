#!/usr/bin/env python3
"""P13-A 256x256 sliding window evaluation with full confusion matrix."""

from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_opening

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P13A_DIR = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_a_multilevel_vit"
sys.path.insert(0, SE)
sys.path.insert(0, P13A_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model import Plan13MultiLevelMFNet  # noqa: E402
from train import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint: str, adapter_bottleneck: int = 32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan13MultiLevelMFNet(
        load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=NUM_CLASSES,
        dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first: {missing[0][:80]}...)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first: {unexpected[0][:80]}...)")
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


def sliding_predict(model, rgb: np.ndarray, dsm: np.ndarray, window=256, stride=128):
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
            pred_sum[y + im:y2 - im, x + jm:x2 - jm] += pred[im:ph - im, jm:pw - jm]
            count[y + im:y2 - im, x + jm:x2 - jm] += 1.0
    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path, no_confusion=False, use_ndsm=False):
    t0 = time.time()
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt_full = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm_raw = np.array(Image.open(dsm_path)).astype(np.float32)
        if use_ndsm:
            ground = grey_opening(dsm_raw, size=101)
            dsm = np.clip((dsm_raw - ground) / 10.0, 0, 1).astype(np.float32)
        else:
            dsm = (dsm_raw - dsm_raw.min()) / max(dsm_raw.max() - dsm_raw.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)

    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) inferring...", end=" ", flush=True)
    pred = sliding_predict(model, rgb, dsm)
    elapsed = time.time() - t0

    mask = gt_full != 255
    gt = gt_full[mask]
    pd = pred[mask]
    total = int(mask.sum())
    correct = int((pd == gt).sum())
    oa = float(correct / max(total, 1) * 100)

    ious, recalls = {}, {}
    counts = {}
    for idx, name in enumerate(CLASS_NAMES):
        pc, lc = pd == idx, gt == idx
        inter = float((pc & lc).sum())
        union = float((pc | lc).sum())
        gt_pixels = float(lc.sum())
        ious[name] = inter / union * 100 if union > 0 else 0.0
        recalls[name] = inter / max(gt_pixels, 1.0) * 100
        counts[f"{name}_inter"] = inter
        counts[f"{name}_union"] = union
        counts[f"{name}_gt"] = gt_pixels

    miou = float(np.mean(list(ious.values())))
    mrecall = float(np.mean(list(recalls.values())))
    print(f"done ({elapsed:.0f}s) OA={oa:.2f}% mIoU={miou:.2f}%")

    result = {
        "tile": tile, "oa": oa, "miou": miou,
        "correct": correct, "total": total,
        **ious, **{f"{k}_recall": v for k, v in recalls.items()}, **counts,
    }

    if not no_confusion:
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for i in range(NUM_CLASSES):
            gt_i = (gt == i)
            for j in range(NUM_CLASSES):
                cm[i, j] = int((gt_i & (pd == j)).sum())
        result["confusion_matrix"] = cm.tolist()

    return result


def print_confusion_matrix(cm, title, row_labels, col_labels, mode="raw"):
    if mode == "raw":
        print(f"\n{title}")
        hdr = "GT \\ Pred  " + "".join(f"{c:>10s}" for c in col_labels)
        print(hdr); print("-" * len(hdr))
        for i, label in enumerate(row_labels):
            print(f"{label:10s}  " + "".join(f"{cm[i,j]:10d}" for j in range(len(col_labels))))
    elif mode == "recall":
        row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
        mat = cm / row_sums * 100
        print(f"\n{title}")
        hdr = "GT \\ Pred  " + "".join(f"{c:>8s}" for c in col_labels)
        print(hdr); print("-" * len(hdr))
        for i, label in enumerate(row_labels):
            print(f"{label:10s}  " + "".join(f"{mat[i,j]:8.1f}" for j in range(len(col_labels))))
    elif mode == "precision":
        col_sums = cm.sum(axis=0, keepdims=True); col_sums[col_sums == 0] = 1
        mat = cm / col_sums * 100
        print(f"\n{title}")
        hdr = "GT \\ Pred  " + "".join(f"{c:>8s}" for c in col_labels)
        print(hdr); print("-" * len(hdr))
        for i, label in enumerate(row_labels):
            print(f"{label:10s}  " + "".join(f"{mat[i,j]:8.1f}" for j in range(len(col_labels))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-confusion", action="store_true",
                        help="Skip confusion matrix computation (for faster eval)")
    parser.add_argument("--ndsm", action="store_true",
                        help="Use nDSM normalization (grey_opening) instead of per-tile min-max")
    args = parser.parse_args()

    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL
    print(f"P13-A eval: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Confusion matrix: {'off' if args.no_confusion else 'on'}")
    print(f"  DSM normalization: {'nDSM' if args.ndsm else 'per-tile min-max'}")
    model, ckpt = load_model(args.checkpoint, args.adapter_bottleneck)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    results = []
    for t in tiles:
        results.append(evaluate_tile(model, t, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths[t],
                                     no_confusion=args.no_confusion, use_ndsm=args.ndsm))

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = float(total_correct / max(total_pixels, 1) * 100)

    per_class_iou, per_class_recall = {}, {}
    for n in CLASS_NAMES:
        inter = sum(r[f"{n}_inter"] for r in results)
        union = sum(r[f"{n}_union"] for r in results)
        gt_pixels = sum(r[f"{n}_gt"] for r in results)
        per_class_iou[n] = float(inter / union * 100) if union > 0 else 0.0
        per_class_recall[n] = float(inter / max(gt_pixels, 1.0) * 100)

    avg_miou = float(np.mean(list(per_class_iou.values())))
    mrecall = float(np.mean(list(per_class_recall.values())))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%  mRecall={mrecall:.2f}%")
    print("  Per-class IoU : " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))
    print("  Per-class Rec : " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items()))

    summary = {
        "avg_oa": avg_oa, "avg_miou": avg_miou,
        "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
        "tiles": results,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_crop_best": ckpt.get("best_v"),
    }

    if not args.no_confusion and any("confusion_matrix" in r for r in results):
        global_cm = sum(np.array(r["confusion_matrix"]) for r in results
                        if "confusion_matrix" in r)
        print_confusion_matrix(global_cm, "CONFUSION MATRIX (raw counts)", CLASS_NAMES, CLASS_NAMES, "raw")
        print_confusion_matrix(global_cm, "ROW-NORMALIZED (% of GT -> Pred, i.e. Recall view)", CLASS_NAMES, CLASS_NAMES, "recall")
        print_confusion_matrix(global_cm, "COLUMN-NORMALIZED (% of Pred <- GT, i.e. Precision view)", CLASS_NAMES, CLASS_NAMES, "precision")

        per_class_precision, per_class_f1 = {}, {}
        for i, name in enumerate(CLASS_NAMES):
            tp = int(global_cm[i, i])
            fp = int(global_cm[:, i].sum() - tp)
            fn = int(global_cm[i, :].sum() - tp)
            per_class_precision[name] = float(tp / max(tp + fp, 1) * 100)
            per_class_f1[name] = float(2 * float(per_class_recall[name]) * float(per_class_precision[name]) /
                                       max(float(per_class_recall[name]) + float(per_class_precision[name]), 1e-8))

        print(f"\n  {'Class':>12s}  {'IoU':>7s}  {'Recall':>7s}  {'Precision':>9s}  {'F1':>7s}")
        print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}")
        for name in CLASS_NAMES:
            print(f"  {name:>12s}  {per_class_iou[name]:6.2f}%  {per_class_recall[name]:6.2f}%  "
                  f"{per_class_precision[name]:8.2f}%  {per_class_f1[name]:6.2f}%")

        summary["confusion_matrix"] = global_cm.tolist()
        summary["confusion_matrix_labels"] = CLASS_NAMES
        summary["per_class_precision"] = per_class_precision
        summary["per_class_f1"] = per_class_f1

    out = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_256_{args.dataset}_plan13_a.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
