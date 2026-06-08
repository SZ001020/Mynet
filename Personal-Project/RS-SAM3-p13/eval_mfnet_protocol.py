#!/usr/bin/env python3
"""MFNet 论文标准评估: 256x256 stride=32, soft-logit 累积, 全局混淆矩阵."""

from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_opening

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
sys.path.insert(0, SE)

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint, model_type, adapter_bottleneck=32):
    if model_type == 'p13a':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_a_multilevel_vit")
        from model import Plan13MultiLevelMFNet as M
    elif model_type == 'p13c':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_c_texture_branch")
        from model import Plan13CTextureMFNet as M
    elif model_type == 'p13e':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_e_building_suppression")
        from model import Plan13EGatedMFNet as M
    elif model_type == 'p13f':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_f_perclass_gate")
        from model import Plan13FPerClassMFNet as M
    else:
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt")
        from model import Plan7PromptMFNet as M

    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})

    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev)

    model = M(sam3, adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
              num_classes=NUM_CLASSES, dropout=0.1, dsm_attn_mode=args.get("dsm_attn_mode", "full"),
              checkpoint_attn=bool(args.get("checkpoint_attn", False)),
              resolution=int(args.get("resolution", 1008))).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt, model_type in ('p13c','p13e','p13f')


def sliding_window_coords(h, w, window=256, stride=32):
    """Generate all window coordinates (x, y, w, h) for sliding window."""
    coords = []
    for y in range(0, h - stride, stride):
        for x in range(0, w - stride, stride):
            x2, y2 = min(x + window, w), min(y + window, h)
            pw, ph = x2 - x, y2 - y
            if ph >= stride and pw >= stride:
                coords.append((y, x, ph, pw))
    return coords


@torch.no_grad()
def eval_mfnet(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path, has_texture, batch_size=4):
    from dataset_adapter import _rgb_to_class
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt_full = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))

    if os.path.exists(dsm_path):
        dsm_raw = np.array(Image.open(dsm_path)).astype(np.float32)
        ground = grey_opening(dsm_raw, size=101)
        dsm = np.clip((dsm_raw - ground) / 10.0, 0, 1).astype(np.float32)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)

    h, w = rgb.shape[:2]
    coords = sliding_window_coords(h, w, 256, 32)
    pred = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)

    t0 = time.time()
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        patches_rgb, patches_dsm = [], []
        for y, x, ph, pw in batch_coords:
            patch = rgb[y:y + ph, x:x + pw]
            dsm_p = dsm[y:y + ph, x:x + pw]
            if ph < 256 or pw < 256:
                patch = np.pad(patch, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                dsm_p = np.pad(dsm_p, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
            patches_rgb.append(patch.transpose(2, 0, 1))
            patches_dsm.append(dsm_p)
        rgb_t = torch.from_numpy(np.stack(patches_rgb)).float().cuda() / 255.0
        dsm_t = torch.from_numpy(np.stack(patches_dsm)).float().cuda()
        if dsm_t.dim() == 3:
            dsm_t = dsm_t.unsqueeze(1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if has_texture:
                refined, _, _ = model(rgb_t, dsm_t)
                logits = refined
            else:
                logits = model(rgb_t, dsm_t)

        logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
        outs = logits.cpu().numpy()

        for (y, x, ph, pw), out in zip(batch_coords, outs):
            im, jm = min(16, ph // 4), min(16, pw // 4)
            pred[y + im:y + ph - im, x + jm:x + pw - jm] += out[:, im:ph - im, jm:pw - jm].transpose(1, 2, 0)

    pred_label = np.argmax(pred, axis=-1)
    elapsed = time.time() - t0

    mask = gt_full != 255
    gt, pd = gt_full[mask], pred_label[mask]
    total = int(mask.sum())
    correct = int((pd == gt).sum())
    oa = correct / max(total, 1) * 100

    ious, recalls, counts = {}, {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        pc, lc = pd == idx, gt == idx
        inter = float((pc & lc).sum()); union = float((pc | lc).sum())
        gt_pix = float(lc.sum())
        ious[name] = inter / union * 100 if union > 0 else 0.0
        recalls[name] = inter / max(gt_pix, 1) * 100
        counts[f"{name}_inter"] = inter
        counts[f"{name}_union"] = union
        counts[f"{name}_gt"] = gt_pix

    miou = float(np.mean(list(ious.values())))
    print(f"  {tile} ({w}x{h}): {len(coords)} patches, {elapsed:.0f}s, OA={oa:.2f}%, mIoU={miou:.2f}%")
    print(f"    IoU: " + " ".join(f"{k}={v:.1f}" for k, v in ious.items()))

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cm[i, j] = int(((gt == i) & (pd == j)).sum())

    return {"tile": tile, "oa": oa, "miou": miou, "correct": correct, "total": total,
            "confusion_matrix": cm, **ious, **{f"{k}_recall": v for k, v in recalls.items()}, **counts}


def load_dataset(dataset):
    from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL
    if dataset == "vaihingen":
        tiles = VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suf, gt_suf = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif'
                     for t in tiles}
    else:
        tiles = POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suf, gt_suf = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_","")}.tif'
                     for t in tiles}
    return tiles, img_dir, gt_dir, img_suf, gt_suf, dsm_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="p11b",
                        choices=["p11b", "p13a", "p13c", "p13e", "p13f"])
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"MFNet Protocol Eval: {args.model_type} on {args.dataset}")
    print(f"  Protocol: 256x256, stride=32, soft-logit accumulation, global CM")

    model, ckpt, has_texture = load_model(args.checkpoint, args.model_type, args.adapter_bottleneck)
    print(f"  Loaded epoch {ckpt.get('epoch')}, crop-best={ckpt.get('best_v', 0):.2f}%")

    tiles, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = load_dataset(args.dataset)
    results = []
    for t in tiles:
        results.append(eval_mfnet(model, t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths[t],
                                  has_texture, args.batch_size))

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = total_correct / max(total_pixels, 1) * 100

    per_class_iou, per_class_recall = {}, {}
    for n in CLASS_NAMES:
        inter = sum(r[f"{n}_inter"] for r in results)
        union = sum(r[f"{n}_union"] for r in results)
        gt_pix = sum(r[f"{n}_gt"] for r in results)
        per_class_iou[n] = inter / union * 100 if union > 0 else 0.0
        per_class_recall[n] = inter / max(gt_pix, 1) * 100

    avg_miou = float(np.mean(list(per_class_iou.values())))

    print("\n" + "=" * 65)
    print(f"MFNet PROTOCOL SUMMARY: {args.model_type} ({args.dataset})")
    print("=" * 65)
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print("  Per-class IoU :  " + "  ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))
    print("  Per-class Rec :  " + "  ".join(f"{k}={v:.2f}" for k, v in per_class_recall.items()))

    global_cm = sum(r["confusion_matrix"] for r in results)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10s}", end="")
    for c in CLASS_NAMES: print(f"{c:>8s}", end="")
    print("\n  " + "-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>8s}", end="")
        for j in range(NUM_CLASSES): print(f"{global_cm[i, j]:8d}", end="")
        print()
    print(f"  {'recall':>8s}", end="")
    for i in range(NUM_CLASSES):
        print(f" {global_cm[i, i] / max(global_cm[i, :].sum(), 1) * 100:6.1f}%", end="")
    print()

    # Key vegetation metrics
    gt = global_cm[2, 3]; tg = global_cm[3, 2]
    grass_gt = global_cm[2, :].sum(); tree_gt = global_cm[3, :].sum()
    print(f"\n  grass->tree: {gt} / {grass_gt} = {gt / max(grass_gt, 1) * 100:.1f}%")
    print(f"  tree->grass: {tg} / {tree_gt} = {tg / max(tree_gt, 1) * 100:.1f}%")
    print(f"  veg sum: {gt + tg}")

    summary = {"avg_oa": avg_oa, "avg_miou": avg_miou,
               "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
               "confusion_matrix": global_cm.tolist(),
               "checkpoint": args.checkpoint, "checkpoint_epoch": ckpt.get("epoch"),
               "protocol": "256x256 stride=32, soft-logit accumulation, global CM (MFNet standard)"}

    out = args.output or os.path.join(os.path.dirname(args.checkpoint),
                                       f"eval_mfnet_{args.dataset}_{args.model_type}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
