#!/usr/bin/env python3
"""Quick 256² evaluation for all Phase 4 checkpoints."""
from __future__ import annotations
import sys, os, json, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
F0_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3"
F1_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion"
F0P_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
METRICS = f"{BASE}/Personal-Project"
sys.path.extend([SE, PHASE1, PHASE_A, F0_DIR, F1_DIR, F0P_DIR, SHARED, METRICS])
from metrics_utils import compute_kappa

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]

def load_sam3():
    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev); return m.cuda()

def load_model(ckpt_path, model_type):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    sam3 = load_sam3()
    if model_type == "f0":
        from model_f0 import MFNetSAM3
        model = MFNetSAM3(sam3, lora_rank=ckpt_args.get("lora_rank", 8),
                          num_classes=5, dropout=0.1, resolution=ckpt_args.get("resolution", 1008)).cuda()
    elif model_type == "f0p":
        from model_f0p import FrozenSAM3DFM
        model = FrozenSAM3DFM(sam3, num_classes=5, dropout=0.1, resolution=ckpt_args.get("resolution", 1008)).cuda()
    elif model_type == "f0p_lora":
        from model_f0p_lora import FrozenSAM3DFMLoRA
        model = FrozenSAM3DFMLoRA(sam3, num_classes=5, dropout=0.1, resolution=ckpt_args.get("resolution", 1008)).cuda()
    elif model_type == "f0p_l2":
        from model_f0p_l2 import FrozenSAM3DFMLoRAv2
        model = FrozenSAM3DFMLoRAv2(sam3, num_classes=5, dropout=0.1, resolution=ckpt_args.get("resolution", 1008)).cuda()
    elif model_type == "f0p_m":
        from model_f0p_m import FrozenSAM3MMLoRA
        model = FrozenSAM3MMLoRA(sam3, num_classes=5, dropout=0.1, resolution=ckpt_args.get("resolution", 1008)).cuda()
    else:  # "f1" or default
        from model_f1 import Phase4AdapterModel
        use_lora = ckpt_args.get("use_lora", False)
        use_prompt = ckpt_args.get("use_prompt", False)
        model = Phase4AdapterModel(sam3, num_classes=5, dropout=0.1,
                                   dsm_attn_mode=ckpt_args.get("dsm_attn_mode", "full"),
                                   checkpoint_attn=ckpt_args.get("checkpoint_attn", False),
                                   resolution=ckpt_args.get("resolution", 1008),
                                   use_lora=use_lora, use_prompt=use_prompt).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model

@torch.no_grad()
def predict_logits(model, patch_rgb, patch_dsm):
    rgb_t = torch.from_numpy(patch_rgb).permute(2,0,1).float().unsqueeze(0)/255.0
    dsm_t = torch.from_numpy(patch_dsm).float().unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(rgb_t.cuda(), dsm_t.cuda())
        logits = F.interpolate(logits.float(), (256, 256), mode="bilinear", align_corners=False)
    return logits[0].cpu().numpy()

def evaluate(model, dataset="vaihingen"):
    from train_f0 import dataset_paths
    _, tiles, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(dataset)
    from dataset_adapter import _rgb_to_class
    all_inter = np.zeros(5, dtype=np.float64)
    all_union = np.zeros(5, dtype=np.float64)
    all_gt = np.zeros(5, dtype=np.float64)
    all_correct, all_total = 0.0, 0.0
    tile_results = []

    for tile in tiles:
        print(f"  {tile}...", flush=True)
        img = np.array(Image.open(f"{img_dir}/{tile}{img_suf}").convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(f"{gt_dir}/{tile}{gt_suf}").convert("RGB")))
        dp = dsm_paths[tile]
        dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2])
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)

        H, W = img.shape[:2]
        logit_sum = np.zeros((5, H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        for y in range(0, H - 128, 128):
            for x in range(0, W - 128, 128):
                y2, x2 = min(y + 256, H), min(x + 256, W)
                ph, pw = y2 - y, x2 - x
                if ph < 128 or pw < 128: continue
                patch = img[y:y2, x:x2]; dsm_p = dsm[y:y2, x:x2]
                if ph < 256 or pw < 256:
                    patch = np.pad(patch, ((0,256-ph),(0,256-pw),(0,0)), mode="reflect")
                    dsm_p = np.pad(dsm_p, ((0,256-ph),(0,256-pw)), mode="reflect")
                logits = predict_logits(model, patch, dsm_p)[:, :ph, :pw]
                trim = min(16, ph//4, pw//4)
                logit_sum[:, y+trim:y2-trim, x+trim:x2-trim] += logits[:, trim:ph-trim, trim:pw-trim]
                count[y+trim:y2-trim, x+trim:x2-trim] += 1.0
        count[count == 0] = 1.0
        logit_sum /= count
        pred = logit_sum.argmax(0)

        mask = gt != 255
        tile_correct = (pred[mask] == gt[mask]).sum()
        tile_total = mask.sum()
        all_correct += tile_correct; all_total += tile_total
        tile_info = {"tile": tile, "oa": float(tile_correct / max(tile_total, 1) * 100)}
        for c in range(5):
            pc, lc = pred == c, gt == c
            inter = (pc & lc).sum()
            union = (pc | lc).sum()
            gt_cnt = lc.sum()
            all_inter[c] += inter; all_union[c] += union; all_gt[c] += gt_cnt
            tile_info[CLASS_NAMES[c]] = float(inter / max(union, 1) * 100)
            tile_info[f"{CLASS_NAMES[c]}_recall"] = float(inter / max(gt_cnt, 1) * 100)
        tile_results.append(tile_info)

    oa = all_correct / max(all_total, 1) * 100
    ious = {CLASS_NAMES[c]: all_inter[c] / max(all_union[c], 1) * 100 for c in range(5)}
    recalls = {CLASS_NAMES[c]: all_inter[c] / max(all_gt[c], 1) * 100 for c in range(5)}
    miou = float(np.mean(list(ious.values())))
    kappa = compute_kappa(all_inter, all_gt, per_class_union=all_union)
    return {"avg_oa": oa, "avg_miou": miou, "kappa": kappa, "per_class_iou": ious, "per_class_recall": recalls, "tiles": tile_results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    args = parser.parse_args()

    configs = [
        # Phase4 main experiments
        ("F0", "/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260518_213923/best_model.pt", "f0"),
        ("F1", "/root/autodl-tmp/runs/plan6_phase4_lora_vaihingen_20260519_124902/best_model.pt", "f1"),
        ("F2", "/root/autodl-tmp/runs/plan6_phase4_frozen_vaihingen_20260519_155635/best_model.pt", "f1"),
        ("F3", "/root/autodl-tmp/runs/plan6_phase4_prompt_vaihingen_20260519_191703/best_model.pt", "f1"),
        # Phase4 supplementary
        ("F0p", "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt", "f0p"),
        ("F0p_L", "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt", "f0p_lora"),
        ("F0p_M", "/root/autodl-tmp/runs/plan6_phase4_f0p_m_vaihingen_20260520_223449/best_model.pt", "f0p_m"),
        ("F0p_L2", "/root/autodl-tmp/runs/plan6_phase4_f0p_l2_vaihingen_20260520_192526/best_model.pt", "f0p_l2"),
        # Unfreeze experiments
        ("U1", "/root/autodl-tmp/runs/plan6_phase4_u24_31_vaihingen_20260525_211145/best_model.pt", "f0p"),
        ("U2", "/root/autodl-tmp/runs/plan6_phase4_u24_31_mlp_vaihingen_20260525_232410/best_model.pt", "f0p"),
        ("U3", "/root/autodl-tmp/runs/plan6_phase4_u28_31_vaihingen_20260526_005659/best_model.pt", "f0p"),
    ]
    for name, ckpt, mtype in configs:
        if not os.path.exists(ckpt):
            print(f"\n=== {name}: SKIP (no checkpoint) ===")
            continue
        print(f"\n=== {name} 256² {args.dataset} eval ===")
        model = load_model(ckpt, mtype)
        r = evaluate(model, args.dataset)
        print(f"  OA={r['avg_oa']:.2f}% mIoU={r['avg_miou']:.2f}% Kappa={r.get('kappa', 0):.4f}")
        print(f"  IoU: {r['per_class_iou']}")
        print(f"  Recall: {r.get('per_class_recall', {})}")
        out_path = os.path.join(os.path.dirname(ckpt), f"eval_256_phase4.json")
        json.dump(r, open(out_path, "w"), indent=2)
        print(f"  Saved: {out_path}")
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
