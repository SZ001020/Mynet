#!/usr/bin/env python3
"""Plan6 Phase 3-D1: 多尺度推理测试增强 [来源: TASAM MS-SAM]

纯推理时改动，零训练成本。对输入做 s× 缩放后分别推理，logit 平均融合。
消除 ViT 16×16 patch grid 和地物的物理对齐偏差。
"""
from __future__ import annotations
import argparse, json, os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"; SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
sys.path.insert(0, SE); sys.path.insert(0, PHASE_DIR)

from dataset_adapter import POTSDAM_VAL, VAIHINGEN_VAL, _rgb_to_class
from model_phase1 import Plan6MMAdapterMFNet
from train_phase1 import dataset_paths, load_sam3
CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def load_model(checkpoint, adapter_bottleneck=32):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    args = ckpt.get("args", {})
    model = Plan6MMAdapterMFNet(load_sam3(),
        adapter_bottleneck=int(args.get("adapter_bottleneck", adapter_bottleneck)),
        num_classes=5, dropout=0.1,
        dsm_attn_mode=args.get("dsm_attn_mode", "full"),
        checkpoint_attn=bool(args.get("checkpoint_attn", False))).cuda()
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval(); return model, ckpt


@torch.no_grad()
def predict_ms(model, rgb, dsm, scales=[0.75, 1.0]):
    """Multi-scale inference: resize input, run model, resize back, average."""
    logits_list = []
    h, w = rgb.shape[-2:]

    for s in scales:
        if s == 1.0:
            logits = model(rgb.cuda(), dsm.cuda())
        else:
            sh, sw = int(h * s), int(w * s)
            r = F.interpolate(rgb, (sh, sw), mode="bilinear", align_corners=False)
            d = F.interpolate(dsm.unsqueeze(1) if dsm.dim()==3 else dsm, (sh, sw),
                             mode="bilinear", align_corners=False)
            if d.dim() == 3: d = d.squeeze(1)
            pad_h = (1008 - sh % 1008) % 1008
            pad_w = (1008 - sw % 1008) % 1008
            if pad_h > 0 or pad_w > 0:
                r = F.pad(r, (0, pad_w, 0, pad_h), mode="reflect")
                d = F.pad(d, (0, pad_w, 0, pad_h), mode="reflect")
            logits = model(r.cuda(), d.cuda())
            logits = logits[..., :sh, :sw]
            if logits.shape[-2:] != (h, w):
                logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
        logits_list.append(logits)

    return torch.stack(logits_list).mean(dim=0)


@torch.no_grad()
def predict_single(model, rgb, dsm):
    return model(rgb.cuda(), dsm.cuda())


def evaluate_tile(model, rgb_np, dsm_np, gt_np, use_ms=False, scales=[0.75, 1.0]):
    H, W = rgb_np.shape[:2]; stride = 128; window = 256
    total_c, total_p = 0, 0
    inter = torch.zeros(5); union = torch.zeros(5); gt_pix = torch.zeros(5)

    for y in range(0, H - stride, stride):
        for x in range(0, W - stride, stride):
            y2, x2 = min(y + window, H), min(x + window, W)
            ph, pw = y2 - y, x2 - x
            if ph < stride or pw < stride: continue
            r = torch.from_numpy(rgb_np[y:y2, x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
            d = torch.from_numpy(dsm_np[y:y2, x:x2]).float().unsqueeze(0)
            r = F.interpolate(r, (1008,1008), mode='bilinear', align_corners=False)
            d = F.interpolate(d.unsqueeze(1), (1008,1008), mode='bilinear', align_corners=False).squeeze(1)

            logits = predict_ms(model, r, d, scales) if use_ms else predict_single(model, r, d)
            logits = F.interpolate(logits, (window, window), mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0, :ph, :pw].cpu().numpy()
            lbl = gt_np[y:y2, x:x2]
            mask = lbl != 255
            total_c += (pred[mask] == lbl[mask]).sum()
            total_p += mask.sum()
            for c in range(5):
                pc, lc = pred == c, lbl == c
                inter[c] += (pc & lc).sum()
                union[c] += (pc | lc).sum()
                gt_pix[c] += lc[mask].sum()

    oa = total_c / max(total_p, 1) * 100
    iou = {CLASS_NAMES[c]: float(inter[c]/union[c].clamp(min=1)*100) for c in range(5)}
    rec = {CLASS_NAMES[c]: float(inter[c]/max(gt_pix[c], 1)*100) for c in range(5)}
    miou = float(np.mean(list(iou.values())))
    print(f"  OA={oa:.2f}% mIoU={miou:.2f}% mRecall={np.mean(list(rec.values())):.2f}%")
    return oa, miou, iou, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="vaihingen")
    parser.add_argument("--ms", action="store_true", help="Enable multi-scale inference")
    parser.add_argument("--scales", default="0.75,1.0", help="Comma-separated scales")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    _, _, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths = dataset_paths(args.dataset)
    tiles = VAIHINGEN_VAL if args.dataset == "vaihingen" else POTSDAM_VAL

    print(f"Plan6 Phase 3-D1: {'MS' if args.ms else 'Single'} inference, scales={scales}")
    model, ckpt = load_model(args.checkpoint)
    print(f"  Loaded epoch {ckpt.get('epoch')} crop-best={ckpt.get('best_v', 0):.2f}%")

    all_inter = torch.zeros(5); all_union = torch.zeros(5); all_gt = torch.zeros(5)
    total_c, total_p = 0, 0
    for tile in tiles:
        ip = f"{img_dir}/{tile}{img_suffix}"
        gp = f"{gt_dir}/{tile}{gt_suffix}"
        dp = dsm_paths[tile]
        rgb = np.array(Image.open(ip).convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
        dsm = np.array(Image.open(dp) if os.path.exists(dp) else np.zeros(rgb.shape[:2])).astype(np.float32)
        if os.path.exists(dp): dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        print(f"  [{tile}]", end=" ", flush=True)
        oa, miou, iou, rec = evaluate_tile(model, rgb, dsm, gt, args.ms, scales)

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
