#!/usr/bin/env python3
"""256² sliding window eval for Phase3 ablation models."""
import argparse, json, os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"; SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
ADIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation"
sys.path.insert(0, SE); sys.path.insert(0, ADIR)
sys.path.insert(0, f"{ADIR}/a0_anchor"); sys.path.insert(0, f"{ADIR}/chain1_adapter")
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter")

from dataset_adapter import VAIHINGEN_VAL, _rgb_to_class
from a0_anchor.model import A0Anchor
from model_a1 import DSMLateFusion
from model_a3 import A3MMAdapterFull

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]
IGNORE_INDEX = 255
RESOLUTION = 1008


def load_sam3():
    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device='cuda')
    os.chdir(prev); return m


@torch.no_grad()
def sliding_eval(model, use_dsm, tiles, img_dir, gt_dir, dsm_paths):
    total_inter = {c: 0.0 for c in CLASS_NAMES}
    total_union = {c: 0.0 for c in CLASS_NAMES}
    total_gt = {c: 0.0 for c in CLASS_NAMES}
    total_correct = 0.0; total_pixels = 0.0

    for tile in tiles:
        ip = f"{img_dir}/{tile}.tif"
        gp = f"{gt_dir}/{tile}.tif"
        dp = dsm_paths[tile]
        img = np.array(Image.open(ip).convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
        dsm = None
        if use_dsm and os.path.exists(dp):
            dsm = np.array(Image.open(dp)).astype(np.float32)
            dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        elif use_dsm:
            dsm = np.zeros(img.shape[:2], dtype=np.float32)

        H, W = img.shape[:2]; stride = 128
        pred_sum = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        for y in range(0, H - 128, stride):
            for x in range(0, W - 128, stride):
                y2, x2 = min(y+256, H), min(x+256, W)
                ph, pw = y2-y, x2-x
                if ph < 128 or pw < 128: continue
                rgb_p = torch.from_numpy(img[y:y2,x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
                rgb_p = F.interpolate(rgb_p, (RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
                if use_dsm:
                    dsm_p = torch.from_numpy(dsm[y:y2,x:x2]).float().unsqueeze(0)
                    dsm_p = F.interpolate(dsm_p.unsqueeze(1), (RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False).squeeze(1)
                with torch.no_grad():
                    logits = model(rgb_p.cuda(), dsm_p.cuda()) if use_dsm else model(rgb_p.cuda())
                logits = F.interpolate(logits, (256,256), mode='bilinear', align_corners=False)
                p = logits.argmax(1)[0,:ph,:pw].cpu().numpy().astype(np.float64)
                m = 16; im, jm = min(m, ph//4), min(m, pw//4)
                pred_sum[y+im:y2-im, x+jm:x2-jm] += p[im:ph-im, jm:pw-jm]
                count[y+im:y2-im, x+jm:x2-jm] += 1.0
        count[count==0] = 1.0
        pred = np.round(pred_sum/count).astype(np.int64)
        mask = gt != IGNORE_INDEX
        total_correct += (pred[mask] == gt[mask]).sum()
        total_pixels += mask.sum()
        for ci, c in enumerate(CLASS_NAMES):
            pc, lc = pred == ci, gt == ci
            total_inter[c] += (pc & lc).sum()
            total_union[c] += (pc | lc).sum()
            total_gt[c] += lc[mask].sum()
        print(f"  {tile} done")

    oa = total_correct / max(total_pixels, 1) * 100
    iou = {c: total_inter[c] / max(total_union[c], 1) * 100 for c in CLASS_NAMES}
    rec = {c: total_inter[c] / max(total_gt[c], 1) * 100 for c in CLASS_NAMES}
    miou = float(np.mean(list(iou.values())))
    mrec = float(np.mean(list(rec.values())))
    return oa, miou, mrec, iou, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True, choices=["a0", "a1", "a3"])
    args = parser.parse_args()

    model_type = args.model_type
    use_dsm = model_type in ("a1", "a3")

    tiles = VAIHINGEN_VAL
    img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
    gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
    dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
    dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif' for t in tiles}

    print(f"Phase3 256² eval: {model_type}")
    sam3 = load_sam3()
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)

    if model_type == "a0":
        model = A0Anchor(sam3, num_classes=5).cuda()
    elif model_type == "a1":
        model = DSMLateFusion(sam3, num_classes=5, dsm_dim=128).cuda()
    else:
        model = A3MMAdapterFull(sam3, num_classes=5).cuda()

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    oa, miou, mrec, iou, rec = sliding_eval(model, use_dsm, tiles, img_dir, gt_dir, dsm_paths)

    print(f"\n256² Results for {model_type}:")
    print(f"  OA={oa:.2f}%  mIoU={miou:.2f}%  mRecall={mrec:.2f}%")
    for c in CLASS_NAMES:
        print(f"  {c}: IoU={iou[c]:.1f}%  Rec={rec[c]:.1f}%")

    out_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_256_vaihingen_{model_type}.json")
    json.dump({"model": model_type, "oa": oa, "miou": miou, "mrecall": mrec,
               "per_class_iou": iou, "per_class_recall": rec}, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    del model, sam3; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
