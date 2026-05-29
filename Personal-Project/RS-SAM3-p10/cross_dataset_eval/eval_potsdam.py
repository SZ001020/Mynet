#!/usr/bin/env python3
"""Plan10: Evaluate Plan7-A / F0 / F0'+L on Potsdam (256² sliding window, soft-logit)."""

import sys, os, json, gc
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
F0_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3"
F0P_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
sys.path.extend([SE, PHASE1, PHASE_A, F0_DIR, F0P_DIR, SHARED])

from dataset_adapter import POTSDAM_VAL, _rgb_to_class
CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NUM_CLASSES = len(CLASS_NAMES)

DATASET = "potsdam"
IMG_DIR = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
GT_DIR = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
DSM_DIR = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
DSM_PATHS = {t: f'{DSM_DIR}/dsm_potsdam_{t.replace("top_potsdam_", "")}.tif' for t in POTSDAM_VAL}
VAL_TILES = POTSDAM_VAL

CHECKPOINTS = {
    "P10-A_Plan7-A": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_potsdam_20260526_210230/best_model.pt",
    "P10-B_F0": "/root/autodl-tmp/runs/plan6_phase4_f0_potsdam_20260527_092853/best_model.pt",
    "P10-C_F0+L": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_potsdam_20260528_105807/best_model.pt",
}


def load_sam3(model: str = "default"):
    import os as _os
    prev = _os.getcwd()
    _os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    _os.chdir(prev)
    return m


def load_model(ckpt_path: str, arch: str):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sam3 = load_sam3()

    if arch == "plan7a":
        from model import Plan7PromptMFNet
        model = Plan7PromptMFNet(sam3, num_classes=NUM_CLASSES, dropout=0.1,
                                 dsm_attn_mode="full", checkpoint_attn=True, resolution=1008).cuda()
        model.load_state_dict(ckpt["model"], strict=False)
    elif arch == "f0":
        from model_f0 import MFNetSAM3
        model = MFNetSAM3(sam3, num_classes=NUM_CLASSES, dropout=0.1, resolution=1008).cuda()
        model.load_state_dict(ckpt["model"], strict=True)
    elif arch == "f0p_lora":
        from model_f0p_lora import FrozenSAM3DFMLoRA
        model = FrozenSAM3DFMLoRA(sam3, num_classes=NUM_CLASSES, dropout=0.1, resolution=1008).cuda()
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, name: str) -> dict:
    ai = np.zeros(NUM_CLASSES, dtype=np.float64)
    au = np.zeros(NUM_CLASSES, dtype=np.float64)
    ac, at = 0.0, 0.0

    for tile in VAL_TILES:
        print(f"  {name} | {tile}...", flush=True)
        img = np.array(Image.open(f"{IMG_DIR}/{tile}_RGB.tif").convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(f"{GT_DIR}/{tile}_label.tif").convert("RGB")))
        dp = DSM_PATHS[tile]
        dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2])
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        H, W = img.shape[:2]

        logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        for y in range(0, H - 128, 128):
            for x in range(0, W - 128, 128):
                y2, x2 = min(y + 256, H), min(x + 256, W)
                ph, pw = y2 - y, x2 - x
                if ph < 128 or pw < 128:
                    continue
                patch = img[y:y2, x:x2]
                dsm_p = dsm[y:y2, x:x2]
                if ph < 256 or pw < 256:
                    patch = np.pad(patch, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                    dsm_p = np.pad(dsm_p, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                rgb_t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                dsm_t = torch.from_numpy(dsm_p).float().unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
                    logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
                logits = logits[0, :, :ph, :pw].detach().cpu().numpy()
                trim = min(16, ph // 4, pw // 4)
                logit_sum[:, y + trim:y2 - trim, x + trim:x2 - trim] += logits[:, trim:ph - trim, trim:pw - trim]
                count[y + trim:y2 - trim, x + trim:x2 - trim] += 1.0

        count[count == 0] = 1.0
        logit_sum /= count
        pred = logit_sum.argmax(0)

        mask = gt != 255
        ac += (pred[mask] == gt[mask]).sum()
        at += mask.sum()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, gt == c
            ai[c] += (pc & lc).sum()
            au[c] += (pc | lc).sum()

    ious = {CLASS_NAMES[c]: float(ai[c] / max(au[c], 1) * 100) for c in range(NUM_CLASSES)}
    oa = float(ac / max(at, 1) * 100)
    miou = float(np.mean(list(ious.values())))
    print(f"    OA={oa:.2f}% mIoU={miou:.2f}%")
    return {"avg_oa": oa, "avg_miou": miou, "per_class_iou": ious}


def main():
    configs = [
        ("P10-A_Plan7-A", CHECKPOINTS["P10-A_Plan7-A"], "plan7a"),
        ("P10-B_F0", CHECKPOINTS["P10-B_F0"], "f0"),
        ("P10-C_F0+L", CHECKPOINTS["P10-C_F0+L"], "f0p_lora"),
    ]

    print(f"Plan10 Potsdam 256² eval ({len(configs)} models, {len(VAL_TILES)} tiles)")
    results = {}

    for name, ckpt, arch in configs:
        print(f"\n--- {name} ---")
        model = load_model(ckpt, arch)
        results[name] = evaluate_model(model, name)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Plan10 Potsdam 256² Summary")
    print(f"{'Model':<20} {'OA':>8} {'mIoU':>8}")
    print(f"{'-'*38}")
    for k, v in results.items():
        print(f"{k:<20} {v['avg_oa']:>7.2f}% {v['avg_miou']:>7.2f}%")
        iou = v.get("per_class_iou", {})
        if iou:
            parts = ", ".join(f"{kk}={vv:.1f}" for kk, vv in iou.items())
            print(f"  {parts}")

    out_path = "/root/autodl-tmp/runs/plan10_potsdam_256eval.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
