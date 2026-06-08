#!/usr/bin/env python3
"""Tile-level soft-logit evaluation for P13-G."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P7_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
P11B_DIR = f"{BASE}/Personal-Project/RS-SAM3-p11/phase_b_ndsm"
P13G_DIR = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_g_strict_veg_texture"
sys.path.insert(0, SE)
sys.path.insert(0, P7_DIR)
sys.path.insert(0, P11B_DIR)
sys.path.insert(0, P13G_DIR)

from dataset_adapter import IGNORE_INDEX, NUM_CLASSES, POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class  # noqa: E402
from dataset_online import compute_ndsm  # noqa: E402

_p7_spec = importlib.util.spec_from_file_location("p7_model", f"{P7_DIR}/model.py")
_p7_mod = importlib.util.module_from_spec(_p7_spec)
assert _p7_spec.loader is not None
_p7_spec.loader.exec_module(_p7_mod)
Plan7PromptMFNet = _p7_mod.Plan7PromptMFNet

_spec = importlib.util.spec_from_file_location("p13g_model", f"{P13G_DIR}/model.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
Plan13GStrictVegTexture = _mod.Plan13GStrictVegTexture

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


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


def build_model(checkpoint: str, adapter_bottleneck: int = 32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt["args"]
    sam3 = load_sam3()
    base = Plan7PromptMFNet(
        sam3,
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=NUM_CLASSES,
        dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=False,
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    base_ckpt = torch.load(args["init_from"], map_location="cuda", weights_only=False)
    base.load_state_dict(base_ckpt["model"], strict=False)
    model = Plan13GStrictVegTexture(
        base,
        resolution=int(args.get("resolution", 1008)),
        use_rgb_texture=not bool(args.get("no_rgb_texture", False)),
        use_ndsm_roughness=bool(args.get("use_ndsm_roughness", False)),
        assert_invariant=True,
    ).cuda()
    model.load_trainable_state_dict(ckpt["head"])
    model.eval()
    return model, ckpt


def load_datasets(dataset: str):
    if dataset == "vaihingen":
        tiles = VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suffix, gt_suffix = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area", "")}.tif' for t in tiles}
    else:
        tiles = POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suffix, gt_suffix = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_", "")}.tif' for t in tiles}
    return tiles, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths


@torch.no_grad()
def predict_patch_soft(model, rgb, dsm, label, mask_mode: str):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dsm_t = torch.from_numpy(dsm).float().unsqueeze(0)
    label_t = torch.from_numpy(label).long().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            rgb_t.cuda(non_blocking=True),
            dsm_t.cuda(non_blocking=True),
            labels=label_t.cuda(non_blocking=True) if mask_mode == "oracle" else None,
            mask_mode=mask_mode,
        )
        logits = F.interpolate(out["final_logits"], (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().float().numpy()


def sliding_predict_soft(model, rgb, dsm, gt_full, mask_mode: str, window=256, stride=128):
    h, w = rgb.shape[:2]
    logit_sum = np.zeros((NUM_CLASSES, h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for y in range(0, h - stride, stride):
        for x in range(0, w - stride, stride):
            y2, x2 = min(y + window, h), min(x + window, w)
            ph, pw = y2 - y, x2 - x
            if ph < stride or pw < stride:
                continue
            rgb_patch = rgb[y:y2, x:x2]
            dsm_patch = dsm[y:y2, x:x2]
            label_patch = gt_full[y:y2, x:x2]
            if ph < window or pw < window:
                rgb_patch = np.pad(rgb_patch, ((0, window - ph), (0, window - pw), (0, 0)), mode="reflect")
                dsm_patch = np.pad(dsm_patch, ((0, window - ph), (0, window - pw)), mode="reflect")
                label_patch = np.pad(label_patch, ((0, window - ph), (0, window - pw)), mode="constant", constant_values=IGNORE_INDEX)
            logits = predict_patch_soft(model, rgb_patch, dsm_patch, label_patch, mask_mode)
            im, jm = min(16, ph // 4), min(16, pw // 4)
            logit_sum[:, y + im:y2 - im, x + jm:x2 - jm] += logits[:, im:ph - im, jm:pw - jm]
            count[y + im:y2 - im, x + jm:x2 - jm] += 1.0
    count[count == 0] = 1.0
    soft = logit_sum / count[np.newaxis, :, :]
    return soft.argmax(0).astype(np.int64)


def evaluate_tile(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path, mask_mode):
    t0 = time.time()
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    gt_full = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm = compute_ndsm(np.array(Image.open(dsm_path)).astype(np.float32))
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)
    print(f"  {tile} {mask_mode} soft-logit...", end=" ", flush=True)
    pred = sliding_predict_soft(model, rgb, dsm, gt_full, mask_mode)
    elapsed = time.time() - t0
    mask = gt_full != IGNORE_INDEX
    gt, pd = gt_full[mask], pred[mask]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cm[i, j] = int(((gt == i) & (pd == j)).sum())
    oa = np.trace(cm) / max(cm.sum(), 1) * 100
    ious, recalls = {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        union = cm[idx, :].sum() + cm[:, idx].sum() - tp
        ious[name] = tp / max(union, 1) * 100
        recalls[name] = tp / max(cm[idx, :].sum(), 1) * 100
    miou = float(np.mean(list(ious.values())))
    print(f"done ({elapsed:.0f}s) OA={oa:.2f}% mIoU={miou:.2f}%")
    return {"tile": tile, "confusion_matrix": cm, "oa": oa, "miou": miou, **ious, **{f"{k}_recall": v for k, v in recalls.items()}}


def summarize(results, checkpoint, mask_mode):
    cm = sum(r["confusion_matrix"] for r in results)
    ious, recalls = {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        union = cm[idx, :].sum() + cm[:, idx].sum() - tp
        ious[name] = tp / max(union, 1) * 100
        recalls[name] = tp / max(cm[idx, :].sum(), 1) * 100
    avg_miou = float(np.mean(list(ious.values())))
    avg_oa = float(np.trace(cm) / max(cm.sum(), 1) * 100)
    gt_err, tg_err = int(cm[2, 3]), int(cm[3, 2])
    return {
        "checkpoint": checkpoint,
        "mask_mode": mask_mode,
        "avg_miou": avg_miou,
        "avg_oa": avg_oa,
        "per_class_iou": ious,
        "per_class_recall": recalls,
        "confusion_matrix": cm.tolist(),
        "grass_to_tree": gt_err,
        "tree_to_grass": tg_err,
        "grass_to_tree_pct": gt_err / max(cm[2, :].sum(), 1) * 100,
        "tree_to_grass_pct": tg_err / max(cm[3, :].sum(), 1) * 100,
        "veg_sum": gt_err + tg_err,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--mask-mode", default="pred", choices=["oracle", "pred"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"P13-G eval: {args.mask_mode} on {args.dataset}")
    model, ckpt = build_model(args.checkpoint)
    print(f"  Loaded epoch={ckpt.get('epoch')} best={ckpt.get('best_v', 0):.2f}%")
    tiles, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = load_datasets(args.dataset)
    results = [evaluate_tile(model, t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths[t], args.mask_mode) for t in tiles]
    summary = summarize(results, args.checkpoint, args.mask_mode)
    print(f"\nSUMMARY {args.mask_mode}: OA={summary['avg_oa']:.2f}% mIoU={summary['avg_miou']:.2f}%")
    print(f"  grass->tree: {summary['grass_to_tree']} ({summary['grass_to_tree_pct']:.1f}%)")
    print(f"  tree->grass: {summary['tree_to_grass']} ({summary['tree_to_grass_pct']:.1f}%)")
    print(f"  veg sum: {summary['veg_sum']}")
    out = args.output or os.path.join(os.path.dirname(args.checkpoint), f"eval_p13g_{args.dataset}_{args.mask_mode}.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
