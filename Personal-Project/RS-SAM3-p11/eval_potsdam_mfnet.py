#!/usr/bin/env python3
"""Batch-wise MFNet protocol eval for Potsdam models.
stride=32, no edge trim, eroded labels, soft-logit, batch processing.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P7 = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
F0D = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3"
F0PD = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline"
sys.path.insert(0, SE); sys.path.insert(0, P7); sys.path.insert(0, F0D); sys.path.insert(0, F0PD)

from dataset_adapter import _rgb_to_class
from model import Plan7PromptMFNet
from model_f0 import MFNetSAM3
from model_f0p_lora import FrozenSAM3DFMLoRA
from train_a import load_sam3

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NC = len(CLASS_NAMES)
TEST_TILES = ["top_potsdam_4_10", "top_potsdam_5_11", "top_potsdam_2_11",
              "top_potsdam_3_10", "top_potsdam_6_11", "top_potsdam_7_12"]
RGB_DIR = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
DSM_DIR = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
GT_DIR = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants_no_Boundary"

MODELS = {
    "P10-A": {"ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_potsdam_20260526_210230/best_model.pt", "type": "plan7"},
    "P10-B": {"ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_potsdam_20260527_092853/best_model.pt", "type": "f0"},
    "P10-C": {"ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_potsdam_20260528_105807/best_model.pt", "type": "f0p_lora"},
}


def make_positions(H, W, stride=32, window=256):
    """Generate all (y, x) positions MFNet-style: no trim, edge-clamped."""
    positions = []
    for y in range(0, H, stride):
        yy = y if y + window <= H else H - window
        for x in range(0, W, stride):
            xx = x if x + window <= W else W - window
            positions.append((yy, xx))
    return positions


def load_model_cfg(cfg):
    ckpt = torch.load(cfg["ckpt"], map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    sam3 = load_sam3()
    t = cfg["type"]
    if t == "plan7":
        m = Plan7PromptMFNet(sam3, adapter_bottleneck=int(args.get("adapter_bottleneck", 32)),
                             num_classes=NC, dropout=0.1,
                             dsm_attn_mode=args.get("dsm_attn_mode", "full"),
                             checkpoint_attn=bool(args.get("checkpoint_attn", False)),
                             resolution=int(args.get("resolution", 1008))).cuda()
    elif t == "f0":
        m = MFNetSAM3(sam3, num_classes=NC, dropout=0.1,
                      lora_rank=int(args.get("lora_rank", 8))).cuda()
    else:  # f0p_lora
        m = FrozenSAM3DFMLoRA(sam3, num_classes=NC, dropout=0.1,
                              lora_rank=int(args.get("lora_rank", 8))).cuda()
    m.load_state_dict(ckpt["model"], strict=False)
    m.eval()
    return m, ckpt


@torch.no_grad()
def eval_tile_batched(model, rgb, dsm, gt, positions, batch_size):
    H, W = rgb.shape[:2]
    logit_sum = np.zeros((NC, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    N = len(positions)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_pos = positions[start:end]
        batch_rgb = np.zeros((len(batch_pos), 256, 256, 3), dtype=np.float32)
        batch_dsm = np.zeros((len(batch_pos), 256, 256), dtype=np.float32)

        for i, (y, x) in enumerate(batch_pos):
            pr = rgb[y:y + 256, x:x + 256]
            pd = dsm[y:y + 256, x:x + 256]
            if pr.shape[0] != 256 or pr.shape[1] != 256:
                ph, pw = 256 - pr.shape[0], 256 - pr.shape[1]
                pr = np.pad(pr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                pd = np.pad(pd, ((0, ph), (0, pw)), mode="reflect")
            batch_rgb[i] = pr
            batch_dsm[i] = pd

        br = torch.from_numpy(batch_rgb).permute(0, 3, 1, 2).cuda() / 255.0
        bd = torch.from_numpy(batch_dsm).unsqueeze(1).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(br, bd)
            logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)

        for i, (y, x) in enumerate(batch_pos):
            logit_sum[:, y:y + 256, x:x + 256] += logits[i].cpu().numpy()
            count[y:y + 256, x:x + 256] += 1.0

    count[count == 0] = 1.0
    logit_sum /= count
    pred = logit_sum.argmax(0).astype(np.int64)

    mask = gt != 255
    g, p = gt[mask], pred[mask]
    correct = int((p == g).sum())
    total = int(mask.sum())
    cm = np.zeros((NC, NC), dtype=np.int64)
    for i in range(NC):
        gi = (g == i)
        for j in range(NC):
            cm[i, j] = int((gi & (p == j)).sum())
    return correct, total, cm


def eval_model(name, cfg, tiles, batch_size):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    model, ckpt = load_model_cfg(cfg)
    print(f"  Loaded: epoch {ckpt.get('epoch')}, crop-best={ckpt.get('best_v', 0):.2f}%")

    global_cm = np.zeros((NC, NC), dtype=np.int64)
    total_c, total_p = 0, 0

    for tile in tiles:
        t0 = time.time()
        ts = tile.replace("top_potsdam_", "")
        rgb = np.array(Image.open(f"{RGB_DIR}/{tile}_RGB.tif").convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(f"{GT_DIR}/{tile}_label_noBoundary.tif").convert("RGB")))
        dsm = np.array(Image.open(f"{DSM_DIR}/dsm_potsdam_{ts}.tif")).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        H, W = rgb.shape[:2]

        positions = make_positions(H, W)
        print(f"  {tile} ({W}x{H}) {len(positions)} patches...", end=" ", flush=True)

        c, t, cm = eval_tile_batched(model, rgb, dsm, gt, positions, batch_size)
        total_c += c; total_p += t; global_cm += cm

        oa_t = c / max(t, 1) * 100
        ious_t = {}
        for i, n in enumerate(CLASS_NAMES):
            tp = int(global_cm[i, i])
            fp_u = int(global_cm[:, i].sum() - tp)
            fn_u = int(global_cm[i, :].sum() - tp)
            ious_t[n] = tp / (tp + fp_u + fn_u) * 100 if tp + fp_u + fn_u > 0 else 0
        print(f"OA={oa_t:.1f}% mIoU={np.mean(list(ious_t.values())):.1f}% ({time.time()-t0:.0f}s)")

    oa = total_c / max(total_p, 1) * 100
    ious, recs = {}, {}
    for i, n in enumerate(CLASS_NAMES):
        tp = int(global_cm[i, i])
        fp_u = int(global_cm[:, i].sum() - tp)
        fn_u = int(global_cm[i, :].sum() - tp)
        ious[n] = round(tp / (tp + fp_u + fn_u) * 100, 2) if tp + fp_u + fn_u > 0 else 0
        recs[n] = round(tp / max(tp + fn_u, 1) * 100, 2)
    miou = np.mean(list(ious.values()))

    print(f"\n  >>> {name}: OA={oa:.2f}%  mIoU={miou:.2f}%")
    print(f"  IoU: {ious}")
    print(f"  Recall: {recs}")
    print(f"  Confusion (recall view):")
    print(f"  {'':>10s}  " + "  ".join(f"{n:>6s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        rs = global_cm[i].sum()
        if rs > 0:
            vals = "  ".join(f"{global_cm[i,j]/rs*100:5.1f}%" for j in range(NC))
            print(f"  {n:>10s}  {vals}")
    return {"name": name, "oa": oa, "miou": miou, "per_class_iou": ious,
            "per_class_recall": recs, "confusion_matrix": global_cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--models", nargs="*", default=["P10-A", "P10-B", "P10-C"])
    args = parser.parse_args()

    print("MFNet Protocol Potsdam Eval (stride=32, no trim, eroded labels, batched)")
    print(f"  Batch size: {args.batch_size}")

    results = {}
    for name in args.models:
        results[name] = eval_model(name, MODELS[name], TEST_TILES, args.batch_size)

    out_path = "/root/autodl-tmp/runs/potsdam_mfnet_protocol_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
