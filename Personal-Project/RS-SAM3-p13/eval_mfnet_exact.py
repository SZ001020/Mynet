#!/usr/bin/env python3
"""MFNet 精确复现评估: 256x256 stride=32, soft-logit 累积, eroded labels, 无边缘裁剪.

与 MFNet train.py test() 完全一致:
1. sliding_window: 边缘对齐 (无裁剪), 全 256x256 patches
2. soft-logit 累积后 argmax
3. eroded labels (无边界像素) 作为 GT
4. 全局混淆矩阵 (6类含clutter), mIoU = mean(IoU[:5])
"""

from __future__ import annotations

import argparse, json, os, sys, time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
sys.path.insert(0, SE)

CLASS_NAMES = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
NUM_CLASSES = 6


def sliding_window_mfnet(h, w, step=32, window_size=256):
    """MFNet exact: edge-aligned, no crop, always full window_size."""
    coords = []
    for x in range(0, h, step):
        if x + window_size > h:
            x = h - window_size
        for y in range(0, w, step):
            if y + window_size > w:
                y = w - window_size
            coords.append((x, y, window_size, window_size))
    return coords


def load_model(checkpoint, model_type, adapter_bottleneck=32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})

    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev)

    if model_type == 'p13g':
        import importlib.util
        _p7_dir = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
        _g_dir = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_g_strict_veg_texture"
        sys.path.insert(0, _p7_dir)
        sys.path.insert(0, _g_dir)
        _p7_spec = importlib.util.spec_from_file_location("p7_model_p13g",
            f"{_p7_dir}/model.py")
        _p7_mod = importlib.util.module_from_spec(_p7_spec)
        _p7_spec.loader.exec_module(_p7_mod)
        Plan7PromptMFNet = _p7_mod.Plan7PromptMFNet
        _g_spec = importlib.util.spec_from_file_location("p13g_model_mfnet",
            f"{_g_dir}/model.py")
        _g_mod = importlib.util.module_from_spec(_g_spec)
        _g_spec.loader.exec_module(_g_mod)
        Plan13GStrictVegTexture = _g_mod.Plan13GStrictVegTexture
        base = Plan7PromptMFNet(sam3, adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
                                num_classes=5, dropout=0.1, dsm_attn_mode=args.get("dsm_attn_mode", "full"),
                                checkpoint_attn=bool(args.get("checkpoint_attn", False)),
                                resolution=int(args.get("resolution", 1008))).cuda()
        base_ckpt = torch.load(args["init_from"], map_location="cuda", weights_only=False)
        base.load_state_dict(base_ckpt["model"], strict=False)
        model = Plan13GStrictVegTexture(base, resolution=int(args.get("resolution", 1008)),
                                        use_rgb_texture=not bool(args.get("no_rgb_texture", False)),
                                        use_ndsm_roughness=bool(args.get("use_ndsm_roughness", False)),
                                        assert_invariant=False).cuda()
        model.load_trainable_state_dict(ckpt["head"])
        model.eval()
        return model, ckpt, 'g'
    elif model_type == 'p13a':
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

    model = M(sam3, adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
              num_classes=5, dropout=0.1, dsm_attn_mode=args.get("dsm_attn_mode", "full"),
              checkpoint_attn=bool(args.get("checkpoint_attn", False)),
              resolution=int(args.get("resolution", 1008))).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt, model_type in ('p13c', 'p13e', 'p13f')


# MFNet palette for eroded labels
_INVERT_PALETTE = {(255,255,255):0, (0,0,255):1, (0,255,255):2, (0,255,0):3, (255,255,0):4, (255,0,0):5}
_VALID_CLASSES = set(_INVERT_PALETTE.values())  # {0,1,2,3,4,5}


def load_eroded_label(tile_name, dataset):
    """Load MFNet-style eroded label. Returns class indices 0-5, 255 for eroded boundary."""
    if dataset == "vaihingen":
        path = f"/root/autodl-tmp/dataset/Vaihingen/gts_eroded_for_participants/{tile_name}_noBoundary.tif"
    else:
        path = f"/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants_no_Boundary/{tile_name}_label_noBoundary.tif"

    rgb = np.array(Image.open(path).convert("RGB"))
    label = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for (r, g, b), idx in _INVERT_PALETTE.items():
        mask = (rgb[:,:,0]==r) & (rgb[:,:,1]==g) & (rgb[:,:,2]==b)
        label[mask] = idx
    return label


@torch.no_grad()
def eval_mfnet_exact(model, tile_name, img_dir, gt_dir, img_suffix, dsm_dir, dataset, has_texture, batch_size=10):
    # Load image (MFNet loads full tile)
    img = np.array(Image.open(f"{img_dir}/{tile_name}{img_suffix}").convert("RGB"))
    img = (1.0 / 255 * img.astype('float32'))

    # Load DSM with nDSM normalization (model was trained on nDSM, not min-max)
    dsm_id = tile_name.replace("top_mosaic_09cm_area", "")
    dsm_path = f"{dsm_dir}/dsm_09cm_matching_area{dsm_id}.tif"
    dsm_raw = np.array(Image.open(dsm_path)).astype(np.float32)
    from scipy.ndimage import grey_opening
    ground = grey_opening(dsm_raw, size=101)
    dsm = np.clip((dsm_raw - ground) / 10.0, 0, 1).astype(np.float32)

    # Load eroded GT label
    gt_e = load_eroded_label(tile_name, dataset)

    h, w = img.shape[:2]
    coords = sliding_window_mfnet(h, w, step=32, window_size=256)
    pred = np.zeros((h, w, 5), dtype=np.float64)  # Our model outputs 5 classes

    t0 = time.time()
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        patches_img, patches_dsm = [], []
        for x, y, _, _ in batch_coords:
            patches_img.append(np.copy(img[x:x+256, y:y+256]).transpose((2, 0, 1)))
            patches_dsm.append(np.copy(dsm[x:x+256, y:y+256]))

        img_t = torch.from_numpy(np.asarray(patches_img)).float().cuda()
        dsm_t = torch.from_numpy(np.asarray(patches_dsm)).float().cuda()
        if dsm_t.dim() == 3:
            dsm_t = dsm_t.unsqueeze(1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if has_texture == 'g':
                result = model(img_t, dsm_t, labels=None, mask_mode="pred")
                outs = result["final_logits"]
            elif has_texture:
                result = model(img_t, dsm_t)
                outs = result[0]  # First return value is refined logits
            else:
                outs = model(img_t, dsm_t)

        outs = F.interpolate(outs.float(), (256, 256), mode="bilinear", align_corners=False)
        outs = outs.detach().cpu().numpy()

        for out, (x, y, _, _) in zip(outs, batch_coords):
            out = out.transpose((1, 2, 0))  # [5, 256, 256] -> [256, 256, 5]
            pred[x:x+256, y:y+256] += out

    pred_label = np.argmax(pred, axis=-1)
    elapsed = time.time() - t0

    # Map: our 0-4 -> MFNet 0-4; clutter(5) not predicted by our model
    # GT: 0-4 = main classes, 5 = clutter (excluded from metrics)
    mask = (gt_e != 255) & (gt_e != 5)  # Exclude clutter and invalid
    gt_masked = gt_e[mask]
    pd_masked = pred_label[mask]

    total = int(mask.sum())
    correct = int((pd_masked == gt_masked).sum())
    oa = correct / max(total, 1) * 100

    cm_full = confusion_matrix(gt_e.ravel(), pred_label.ravel(), labels=list(range(5)))
    cm = cm_full[:5, :5]  # 5 main classes only

    ious = {}
    for i in range(5):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        ious[CLASS_NAMES[i]] = tp / max(tp + fp + fn, 1) * 100

    miou = float(np.nanmean(list(ious.values())))
    print(f"  {tile_name} ({w}x{h}): {len(coords)} patches, {elapsed:.0f}s, OA={oa:.2f}%, mIoU={miou:.2f}%")
    print(f"    IoU: " + " ".join(f"{k}={v:.1f}" for k, v in ious.items()))

    return {"tile": tile_name, "oa": oa, "miou": miou, "correct": correct, "total": total,
            "confusion_matrix": cm, "ious": ious}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="p11b", choices=["p11b","p13a","p13c","p13e","p13f","p13g"])
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen","potsdam"])
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    print(f"MFNet EXACT Eval: {args.model_type} on {args.dataset}")
    print(f"  Protocol: 256x256 stride=32, no edge crop, eroded labels, soft-logit, global CM")
    print(f"  mIoU: mean IoU over 5 main classes (excl. clutter)")

    model, ckpt, has_texture = load_model(args.checkpoint, args.model_type)

    if args.dataset == "vaihingen":
        from dataset_adapter import VAIHINGEN_VAL as tiles
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suf = ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"

    results = []
    for tile_name in tiles:
        results.append(eval_mfnet_exact(model, tile_name, img_dir, gt_dir, img_suf,
                                        dsm_dir, args.dataset, has_texture, args.batch_size))

    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    avg_oa = total_correct / max(total_pixels, 1) * 100

    # Global confusion matrix
    global_cm = sum(r["confusion_matrix"] for r in results)
    per_class_iou = {}
    for i in range(5):
        tp = global_cm[i, i]
        fp = global_cm[:, i].sum() - tp
        fn = global_cm[i, :].sum() - tp
        per_class_iou[CLASS_NAMES[i]] = tp / max(tp + fp + fn, 1) * 100
    avg_miou = float(np.nanmean(list(per_class_iou.values())))

    print("\n" + "=" * 65)
    print(f"MFNet EXACT PROTOCOL: {args.model_type} ({args.dataset})")
    print("=" * 65)
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print("  Per-class IoU:  " + "  ".join(f"{k}={v:.2f}" for k, v in per_class_iou.items()))

    # Confusion matrix
    print(f"\n  CM (5 classes, excl. clutter):")
    print(f"  {'':>10s}", end="")
    for c in CLASS_NAMES[:5]: print(f"{c:>8s}", end="")
    print("\n  " + "-" * 50)
    for i in range(5):
        print(f"  {CLASS_NAMES[i]:>10s}", end="")
        for j in range(5): print(f"{global_cm[i, j]:8d}", end="")
        print()
    print(f"  {'recall':>10s}", end="")
    for i in range(5):
        print(f" {global_cm[i,i]/max(global_cm[i,:].sum(),1)*100:6.1f}%", end="")
    print()

    gt = global_cm[2, 3]; tg = global_cm[3, 2]
    print(f"\n  grass->tree: {gt} ({gt/global_cm[2,:].sum()*100:.1f}%)")
    print(f"  tree->grass: {tg} ({tg/global_cm[3,:].sum()*100:.1f}%)")

    summary = {"avg_oa": avg_oa, "avg_miou": avg_miou, "per_class_iou": per_class_iou,
               "confusion_matrix": global_cm.tolist(),
               "checkpoint": args.checkpoint,
               "protocol": "MFNet exact: 256x256 stride=32, no edge crop, eroded labels, soft-logit accumulation"}
    out = os.path.join(os.path.dirname(args.checkpoint), f"eval_mfnet_exact_{args.dataset}_{args.model_type}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
