#!/usr/bin/env python3
"""MFNet 标准协议评估: 256x256 stride=128, soft-logit 累积 + 全局混淆矩阵.

与 per-patch argmax 不同，soft-logit 累积保留了边界区域的不确定性，
通常比 per-patch argmax 高 0.4-0.8pp mIoU。
"""

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


def load_model_and_data(checkpoint, model_type, adapter_bottleneck=32):
    """Load model based on type ('p11b','p13a','p13c','p13e','p13f')."""
    import importlib

    if model_type == 'p13a':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_a_multilevel_vit")
        from model import Plan13MultiLevelMFNet as ModelClass
    elif model_type == 'p13c':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_c_texture_branch")
        from model import Plan13CTextureMFNet as ModelClass
    elif model_type == 'p13e':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_e_building_suppression")
        from model import Plan13EGatedMFNet as ModelClass
    elif model_type == 'p13f':
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p13/phase_f_perclass_gate")
        from model import Plan13FPerClassMFNet as ModelClass
    else:
        sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt")
        from model import Plan7PromptMFNet as ModelClass

    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})

    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev)

    model = ModelClass(
        sam3,
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=NUM_CLASSES, dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False)),
        resolution=int(args.get("resolution", 1008)),
    ).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt


def load_datasets(dataset):
    from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL
    if dataset == "vaihingen":
        tiles = VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suffix, gt_suffix = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif'
                     for t in tiles}
    else:
        tiles = POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suffix, gt_suffix = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_","")}.tif'
                     for t in tiles}
    return tiles, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths


@torch.no_grad()
def predict_patch_soft(model, rgb, dsm, model_type):
    """Return soft logits (not argmax) for soft-logit accumulation."""
    rgb_t = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0)/255.0
    dsm_t = torch.from_numpy(dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if model_type in ('p13c','p13e','p13f'):
            refined, _, _ = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        else:
            refined = model(rgb_t.cuda(non_blocking=True), dsm_t.cuda(non_blocking=True))
        logits = F.interpolate(refined, (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().float().numpy()  # [C, H, W]


def sliding_predict_soft(model, rgb, dsm, model_type, window=256, stride=128):
    """Soft-logit accumulation sliding window."""
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
            if ph < window or pw < window:
                rgb_patch = np.pad(rgb_patch, ((0,window-ph),(0,window-pw),(0,0)), mode="reflect")
                dsm_patch = np.pad(dsm_patch, ((0,window-ph),(0,window-pw)), mode="reflect")
            logits = predict_patch_soft(model, rgb_patch, dsm_patch, model_type)
            im, jm = min(16, ph//4), min(16, pw//4)
            logit_sum[:, y+im:y2-im, x+jm:x2-jm] += logits[:, im:ph-im, jm:pw-jm]
            count[y+im:y2-im, x+jm:x2-jm] += 1.0

    count[count==0] = 1.0
    soft_logit = logit_sum / count[np.newaxis, :, :]
    return soft_logit.argmax(0).astype(np.int64), soft_logit


def evaluate_tile(model, tile, img_dir, gt_dir, img_suffix, gt_suffix, dsm_path, model_type, use_ndsm=True):
    t0 = time.time()
    rgb = np.array(Image.open(f"{img_dir}/{tile}{img_suffix}").convert("RGB"))
    from dataset_adapter import _rgb_to_class
    gt_full = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suffix}").convert("RGB")))
    if os.path.exists(dsm_path):
        dsm_raw = np.array(Image.open(dsm_path)).astype(np.float32)
        if use_ndsm:
            ground = grey_opening(dsm_raw, size=101)
            dsm = np.clip((dsm_raw-ground)/10.0, 0, 1).astype(np.float32)
        else:
            dsm = (dsm_raw-dsm_raw.min())/max(dsm_raw.max()-dsm_raw.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)

    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) soft-logit...", end=" ", flush=True)
    pred, _soft = sliding_predict_soft(model, rgb, dsm, model_type)
    elapsed = time.time() - t0

    mask = gt_full != 255
    gt, pd = gt_full[mask], pred[mask]
    total = int(mask.sum())
    correct = int((pd==gt).sum())
    oa = correct/max(total,1)*100

    ious, recalls, counts = {}, {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        pc, lc = pd==idx, gt==idx
        inter = float((pc&lc).sum()); union = float((pc|lc).sum())
        gt_pix = float(lc.sum())
        ious[name] = inter/union*100 if union>0 else 0.0
        recalls[name] = inter/max(gt_pix,1)*100
        counts[f"{name}_inter"] = inter
        counts[f"{name}_union"] = union
        counts[f"{name}_gt"] = gt_pix

    miou = float(np.mean(list(ious.values())))
    print(f"done ({elapsed:.0f}s) OA={oa:.2f}% mIoU={miou:.2f}%")

    cm = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cm[i,j] = int(((gt==i)&(pd==j)).sum())

    return {"tile": tile, "oa": oa, "miou": miou, "correct": correct, "total": total,
            "confusion_matrix": cm, **ious, **{f"{k}_recall":v for k,v in recalls.items()}, **counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="p11b",
                        choices=["p11b","p13a","p13c","p13e","p13f"])
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen","potsdam"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt")

    print(f"MFNet Standard Eval: {args.model_type} on {args.dataset}")
    print(f"  Protocol: 256x256 stride=128, soft-logit accumulation, global confusion matrix")

    model, ckpt = load_model_and_data(args.checkpoint, args.model_type, args.adapter_bottleneck)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v',0):.2f}%")

    tiles, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = load_datasets(args.dataset)
    results = [evaluate_tile(model, t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths[t],
                             args.model_type) for t in tiles]

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = total_correct/max(total_pixels,1)*100

    per_class_iou, per_class_recall = {}, {}
    for n in CLASS_NAMES:
        inter = sum(r[f"{n}_inter"] for r in results)
        union = sum(r[f"{n}_union"] for r in results)
        gt_pix = sum(r[f"{n}_gt"] for r in results)
        per_class_iou[n] = inter/union*100 if union>0 else 0.0
        per_class_recall[n] = inter/max(gt_pix,1)*100

    avg_miou = float(np.mean(list(per_class_iou.values())))

    print("\n" + "=" * 60)
    print(f"SOFT-LOGIT SUMMARY: {args.model_type}")
    print("=" * 60)
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print("  Per-class IoU : " + ", ".join(f"{k}={v:.2f}" for k,v in per_class_iou.items()))
    print("  Per-class Rec : " + ", ".join(f"{k}={v:.2f}" for k,v in per_class_recall.items()))

    # Global confusion matrix
    global_cm = sum(r["confusion_matrix"] for r in results)
    print(f"\n{'GT\\Pred':>10s}", end="")
    for c in CLASS_NAMES: print(f"{c:>8s}", end="")
    print("\n" + "-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>10s}", end="")
        for j in range(NUM_CLASSES): print(f"{global_cm[i,j]:8d}", end="")
        print()
    print(f"{'recall%':>10s}", end="")
    for i in range(NUM_CLASSES):
        print(f"{global_cm[i,i]/max(global_cm[i,:].sum(),1)*100:7.1f}%", end="")
    print()

    # grass->tree, tree->grass
    gt = global_cm[2,3]; tg = global_cm[3,2]
    print(f"\n  grass->tree: {gt} ({gt/global_cm[2,:].sum()*100:.1f}%)")
    print(f"  tree->grass: {tg} ({tg/global_cm[3,:].sum()*100:.1f}%)")
    print(f"  veg sum: {gt+tg}")

    summary = {"avg_oa": avg_oa, "avg_miou": avg_miou,
               "per_class_iou": per_class_iou, "per_class_recall": per_class_recall,
               "confusion_matrix": global_cm.tolist(),
               "checkpoint": args.checkpoint, "checkpoint_epoch": ckpt.get("epoch"),
               "protocol": "256² sliding window, stride=128, soft-logit accumulation"}

    out = args.output or os.path.join(os.path.dirname(args.checkpoint),
                                       f"eval_soft_{args.dataset}_{args.model_type}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
