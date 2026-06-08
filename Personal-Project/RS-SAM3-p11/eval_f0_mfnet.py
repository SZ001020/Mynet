#!/usr/bin/env python3
"""Eval Phase4 F0 with MFNet protocol: stride=32, no trim, eroded labels."""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
sys.path.insert(0, f"{BASE}/Reference-Project/SegEarth-OV-3-main")
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt")
from dataset_adapter import VAIHINGEN_VAL, _rgb_to_class
from train_a import dataset_paths, load_sam3
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3")
from model_f0 import MFNetSAM3

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
NC = len(CLASS_NAMES)

ckpt = torch.load("/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260518_213923/best_model.pt",
                  map_location="cuda", weights_only=False)
margs = ckpt.get("args", {})
sam3 = load_sam3()
model = MFNetSAM3(sam3, num_classes=NC, dropout=0.1,
                  lora_rank=int(margs.get("lora_rank", 8))).cuda()
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"F0 loaded: epoch {ckpt.get('epoch')}, crop-best={ckpt.get('best_v', 0):.2f}%")

_, _, img_dir, _, _, _, dsm_paths = dataset_paths("vaihingen")
tiles = VAIHINGEN_VAL
eroded_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_eroded_for_participants"


def pad_to_size(img, th, tw):
    h, w = img.shape[:2]
    if h >= th and w >= tw:
        return img
    ph, pw = max(0, th - h), max(0, tw - w)
    if img.ndim == 3:
        return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return np.pad(img, ((0, ph), (0, pw)), mode="reflect")


@torch.no_grad()
def pred_patch(model, rgb, dsm):
    rt = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dt = torch.from_numpy(dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        lo = model(rt.cuda(), dt.cuda())
        lo = F.interpolate(lo.float(), (256, 256), mode="bilinear", align_corners=False)
    return lo[0].cpu().numpy()


def sliding_mfnet(model, img_np, dsm_np, stride=32):
    H, W = img_np.shape[:2]
    ws = 256
    ls = np.zeros((NC, H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for y in range(0, H, stride):
        if y + ws > H:
            y = H - ws
        for x in range(0, W, stride):
            if x + ws > W:
                x = W - ws
            pr = img_np[y:y + ws, x:x + ws]
            pd = dsm_np[y:y + ws, x:x + ws]
            if pr.shape[0] != ws or pr.shape[1] != ws:
                pr = pad_to_size(pr, ws, ws)
                pd = pad_to_size(pd, ws, ws)
            lo = pred_patch(model, pr, pd)
            ls[:, y:y + ws, x:x + ws] += lo
            cnt[y:y + ws, x:x + ws] += 1.0
    cnt[cnt == 0] = 1.0
    ls /= cnt
    return ls.argmax(0).astype(np.int64)


global_cm = np.zeros((NC, NC), dtype=np.int64)
total_c, total_p = 0, 0
for tile in tiles:
    rgb = np.array(Image.open(f"{img_dir}/{tile}.tif").convert("RGB"))
    gt = _rgb_to_class(np.array(Image.open(f"{eroded_dir}/{tile}_noBoundary.tif").convert("RGB")))
    dp = dsm_paths[tile]
    if os.path.exists(dp):
        dsm = np.array(Image.open(dp)).astype(np.float32)
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    else:
        dsm = np.zeros(rgb.shape[:2], dtype=np.float32)
    print(f"  {tile} ({rgb.shape[1]}x{rgb.shape[0]}) ...", end=" ", flush=True)
    pred = sliding_mfnet(model, rgb, dsm, stride=32)
    mask = gt != 255
    g, p = gt[mask], pred[mask]
    total_c += int((p == g).sum())
    total_p += int(mask.sum())
    for i in range(NC):
        gi = (g == i)
        for j in range(NC):
            global_cm[i, j] += int((gi & (p == j)).sum())
    oa = total_c / max(total_p, 1) * 100
    ious, recs = {}, {}
    for i, n in enumerate(CLASS_NAMES):
        tp = int(global_cm[i, i])
        fp = int(global_cm[:, i].sum() - tp)
        fn = int(global_cm[i, :].sum() - tp)
        ious[n] = tp / (tp + fp + fn) * 100 if tp + fp + fn > 0 else 0
        recs[n] = tp / max(tp + fn, 1) * 100
    print(f"OA={oa:.2f}% mIoU={np.mean(list(ious.values())):.2f}%")

miou = np.mean(list(ious.values()))
print(f"\nF0 MFNet Protocol: OA={oa:.2f}% mIoU={miou:.2f}%")
print(f"  IoU: {json.dumps({k: round(v, 2) for k, v in ious.items()})}")
print(f"  Recall: {json.dumps({k: round(v, 2) for k, v in recs.items()})}")
