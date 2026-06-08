#!/usr/bin/env python3
"""Evaluate with MFNet original protocol: stride=32, no edge trim, eroded labels."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_DIR)

from dataset_adapter import VAIHINGEN_VAL, _rgb_to_class
from model import Plan7PromptMFNet
from train_a import dataset_paths, load_sam3

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan7PromptMFNet(
        load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", 32)),
        num_classes=NUM_CLASSES, dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt


def pad_to_size(img, target_h, target_w):
    h, w = img.shape[:2]
    if h >= target_h and w >= target_w:
        return img
    pad_h = max(0, target_h - h); pad_w = max(0, target_w - w)
    if img.ndim == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")


@torch.no_grad()
def predict_patch_logits(model, rgb_patch, dsm_patch):
    rgb_t = torch.from_numpy(rgb_patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm_patch).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(), dsm_t.cuda())
        logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().numpy()


def sliding_mfnet_protocol(model, img_np, dsm_np, stride=32):
    """MFNet protocol: stride=32, no edge trim, soft-logit accumulation."""
    H, W = img_np.shape[:2]
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    window = 256

    for y in range(0, H, stride):
        if y + window > H:
            y = H - window
        for x in range(0, W, stride):
            if x + window > W:
                x = W - window
            patch_rgb = img_np[y:y+window, x:x+window]
            patch_dsm = dsm_np[y:y+window, x:x+window]
            if patch_rgb.shape[0] != window or patch_rgb.shape[1] != window:
                patch_rgb = pad_to_size(patch_rgb, window, window)
                patch_dsm = pad_to_size(patch_dsm, window, window)

            logits = predict_patch_logits(model, patch_rgb, patch_dsm)
            logit_sum[:, y:y+window, x:x+window] += logits
            count[y:y+window, x:x+window] += 1.0

    count[count == 0] = 1.0
    logit_sum /= count
    return logit_sum.argmax(0).astype(np.int64)


def evaluate_tile(model, tile, img_dir, dsm_path, eroded_gt_dir):
    rgb = np.array(Image.open(f"{img_dir}/{tile}.tif").convert("RGB"))

    # Eroded labels (MFNet protocol)
    tile_num = tile.replace("top_mosaic_09cm_area", "")
    gt_path = f"{eroded_gt_dir}/{tile}_noBoundary.tif"
    gt_full = _rgb_to_class(np.array(Image.open(gt_path).convert("RGB")))

    # DSM: per-tile min-max (MFNet protocol)
    if os.path.exists(dsm_path):
        dsm = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)

    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) stride=32...", end=" ", flush=True)
    pred = sliding_mfnet_protocol(model, rgb, dsm, stride=32)

    mask = gt_full != 255
    gt = gt_full[mask]; pd = pred[mask]
    total = int(mask.sum())
    correct = int((pd == gt).sum())
    oa = float(correct / max(total, 1) * 100)

    ious, recalls = {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        pc, lc = pd == idx, gt == idx
        inter = float((pc & lc).sum())
        union = float((pc | lc).sum())
        gt_pixels = float(lc.sum())
        ious[name] = inter / union * 100 if union > 0 else 0.0
        recalls[name] = inter / max(gt_pixels, 1.0) * 100

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        gt_i = (gt == i)
        for j in range(NUM_CLASSES):
            cm[i, j] = int((gt_i & (pd == j)).sum())

    miou = float(np.mean(list(ious.values())))
    print(f"OA={oa:.2f}% mIoU={miou:.2f}%")
    return {"tile": tile, "oa": oa, "miou": miou, "correct": correct, "total": total,
            "per_class_iou": ious, "per_class_recall": recalls, "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, _, img_dir, _, _, _, dsm_paths = dataset_paths("vaihingen")
    tiles = VAIHINGEN_VAL
    eroded_gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_eroded_for_participants"

    print(f"MFNet Protocol Eval (stride=32, no trim, eroded labels)")
    print(f"  Checkpoint: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint)
    print(f"  Epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    results = [evaluate_tile(model, t, img_dir, dsm_paths[t], eroded_gt_dir) for t in tiles]

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = float(total_correct / max(total_pixels, 1) * 100)

    per_class_iou, per_class_recall = {}, {}
    global_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for r in results:
        global_cm += np.array(r["confusion_matrix"])

    for i, name in enumerate(CLASS_NAMES):
        tp = int(global_cm[i, i])
        fp = int(global_cm[:, i].sum() - tp)
        fn = int(global_cm[i, :].sum() - tp)
        union = tp + fp + fn
        per_class_iou[name] = float(tp / union * 100) if union > 0 else 0.0
        per_class_recall[name] = float(tp / max(tp + fn, 1) * 100)

    avg_miou = float(np.mean(list(per_class_iou.values())))

    print(f"\n{'='*60}")
    print(f"MFNet PROTOCOL RESULTS (stride=32, no trim, eroded labels)")
    print(f"{'='*60}")
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print(f"  Per-class IoU: {json.dumps({k: round(v,2) for k,v in per_class_iou.items()})}")
    print(f"  Per-class Rec: {json.dumps({k: round(v,2) for k,v in per_class_recall.items()})}")

    # Row-normalized confusion
    print(f"\n  Row-Normalized Confusion (Recall view):")
    print(f"  {'':>10s}  " + "  ".join(f"{n:>6s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row_sum = global_cm[i].sum()
        if row_sum > 0:
            vals = "  ".join(f"{global_cm[i,j]/row_sum*100:5.1f}%" for j in range(5))
            print(f"  {name:>10s}  {vals}")

    summary = {
        "protocol": "MFNet original: stride=32, no edge trim, eroded labels, per-tile min-max DSM, soft-logit",
        "checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
        "avg_oa": avg_oa, "avg_miou": avg_miou,
        "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
        "confusion_matrix": global_cm.tolist(), "labels": CLASS_NAMES,
        "per_tile": results,
    }
    out = args.output or os.path.join(os.path.dirname(args.checkpoint), "eval_mfnet_protocol.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
