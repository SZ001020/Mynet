#!/usr/bin/env python3
"""Plan6 P2-D1: Multi-scale TTA on Plan6 Phase1 best checkpoint.

Same multi-scale eval as Plan7-D1, applied to Plan6's pure MMAdapter
(no DSM prompt branch). Tests whether patch-grid alignment bias is
independent of Plan7's prompt architecture.
"""

from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
sys.path.insert(0, SE)
sys.path.insert(0, PHASE_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from model_phase1 import Plan6MMAdapterMFNet  # noqa: E402
from train_phase1 import dataset_paths, load_sam3  # noqa: E402

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan6MMAdapterMFNet(
        load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", 32)),
        num_classes=NUM_CLASSES,
        dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "adapter"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch_logits(model, rgb_patch: np.ndarray, dsm_patch: np.ndarray) -> np.ndarray:
    rgb_t = torch.from_numpy(rgb_patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm_patch).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().numpy()


def sliding_logits_at_scale(model, img_np, dsm_np, scale, window=256, stride=128):
    H, W = img_np.shape[:2]
    if scale != 1.0:
        new_h, new_w = int(H * scale), int(W * scale)
        img = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))
        dsm = np.array(Image.fromarray(dsm_np).resize((new_w, new_h), Image.BILINEAR))
    else:
        new_h, new_w = H, W
        img, dsm = img_np, dsm_np

    scaled_stride = max(32, int(stride * scale))
    logit_sum = np.zeros((NUM_CLASSES, new_h, new_w), dtype=np.float64)
    count = np.zeros((new_h, new_w), dtype=np.float64)

    for y in range(0, new_h - scaled_stride, scaled_stride):
        for x in range(0, new_w - scaled_stride, scaled_stride):
            y2, x2 = min(y + window, new_h), min(x + window, new_w)
            ph, pw = y2 - y, x2 - x
            if ph < scaled_stride or pw < scaled_stride:
                continue
            patch_rgb, patch_dsm = img[y:y2, x:x2], dsm[y:y2, x:x2]
            if ph < window or pw < window:
                pad_h, pad_w = max(0, window - ph), max(0, window - pw)
                patch_rgb = np.pad(patch_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                patch_dsm = np.pad(patch_dsm, ((0, pad_h), (0, pad_w)), mode="reflect")
            logits = predict_patch_logits(model, patch_rgb, patch_dsm)[:, :ph, :pw]
            trim = min(16, ph // 4, pw // 4)
            yy, xx = y + trim, x + trim
            yy2, xx2 = y2 - trim, x2 - trim
            if yy2 > yy and xx2 > xx:
                logit_sum[:, yy:yy2, xx:xx2] += logits[:, trim:ph - trim, trim:pw - trim]
                count[yy:yy2, xx:xx2] += 1.0

    count[count == 0] = 1.0
    logit_sum /= count
    if scale != 1.0:
        logit_t = torch.from_numpy(logit_sum).unsqueeze(0)
        logit_t = F.interpolate(logit_t, (H, W), mode="bilinear", align_corners=False)
        logit_sum = logit_t[0].numpy()
    return logit_sum


@torch.no_grad()
def sliding_predict_multiscale(model, img_np, dsm_np, scales, window=256, stride=128):
    H, W = img_np.shape[:2]
    logit_acc = np.zeros((NUM_CLASSES, H, W), dtype=np.float64)
    for scale in scales:
        print(f"    scale {scale:.2f}×...", end=" ", flush=True)
        logit_acc += sliding_logits_at_scale(model, img_np, dsm_np, scale, window, stride)
        print("done")
    logit_acc /= len(scales)
    return logit_acc.argmax(0).astype(np.int64)


def evaluate_tile_ms(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path, scales):
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)
    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) scales={scales}:", flush=True)
    pred = sliding_predict_multiscale(model, rgb, dsm, scales=scales)
    mask = gt != 255
    total = int(mask.sum())
    correct = int((pred[mask] == gt[mask]).sum())
    oa = float(correct / max(total, 1) * 100)
    ious, recalls, counts = {}, {}, {}
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
    return {"tile": tile, "oa": oa, "miou": miou, "correct": correct, "total": total,
            **ious, **{f"{k}_recall": v for k, v in recalls.items()}, **counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL
    print(f"Plan6 P2-D1 multi-scale eval: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Scales: {args.scales}")
    model, ckpt = load_model(args.checkpoint)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    results = [evaluate_tile_ms(model, t, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths[t], tuple(args.scales)) for t in tiles]
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

    print("\n=== P2-D1 Multi-Scale 256² Accumulated ===")
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%  mRecall={mrecall:.2f}%")
    print("  Per-class IoU : " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))
    print("  Per-class Rec : " + ", ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items()))

    out_dir = args.output or os.path.dirname(args.checkpoint)
    tag = "_".join(f"{s:.2f}".replace(".", "_") for s in args.scales)
    out_path = os.path.join(out_dir, f"eval_p2d1_ms_{args.dataset}_{tag}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"scales": list(args.scales), "avg_oa": avg_oa, "avg_miou": avg_miou,
                    "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
                    "tiles": results, "checkpoint": args.checkpoint}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
