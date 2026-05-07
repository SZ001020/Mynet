#!/usr/bin/env python3
"""
MFNet 评估协议复现：256² 滑动窗口 + overlap 平均
"""

import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/RS-SAM3-p3r')
from adapter_unet import AdapterSAM3UNetFormer, AdapterSAM3UNetFormerDSM
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']


def load_rgb_model(ckpt_path, dropout=0.1):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    model = AdapterSAM3UNetFormer(sam3, adapter_bottleneck=32, num_classes=5, dropout=dropout).cuda()
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt


def load_dsm_model(ckpt_path):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    model = AdapterSAM3UNetFormerDSM(sam3, adapter_bottleneck=32, num_classes=5, dropout=0.1).cuda()
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch_rgb(model, patch_256):
    """256² RGB → resize 1008 → model → resize 256 → pred"""
    t = torch.from_numpy(patch_256).permute(2,0,1).float().unsqueeze(0)/255.0
    t = F.interpolate(t, (1008,1008), mode='bilinear', align_corners=False)
    logits = model(t.cuda())
    logits = F.interpolate(logits, (256,256), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


@torch.no_grad()
def predict_patch_dsm(model, patch_rgb_256, patch_dsm_256):
    """256² RGB+DSM → resize 1008 → model → resize 256 → pred"""
    t_rgb = torch.from_numpy(patch_rgb_256).permute(2,0,1).float().unsqueeze(0)/255.0
    t_dsm = torch.from_numpy(patch_dsm_256).float().unsqueeze(0)
    t_rgb = F.interpolate(t_rgb, (1008,1008), mode='bilinear', align_corners=False)
    t_dsm = F.interpolate(t_dsm.unsqueeze(1), (1008,1008), mode='bilinear', align_corners=False)
    logits = model(t_rgb.cuda(), t_dsm.cuda())
    logits = F.interpolate(logits, (256,256), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def sliding_256(tile_h, tile_w, model, img_np, dsm_np=None, stride=128):
    """256² sliding window with overlap averaging."""
    pred_sum = np.zeros((tile_h, tile_w), dtype=np.float64)
    count = np.zeros((tile_h, tile_w), dtype=np.float64)

    for y in range(0, tile_h - 128, stride):
        for x in range(0, tile_w - 128, stride):
            y2, x2 = min(y + 256, tile_h), min(x + 256, tile_w)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue

            patch_rgb = img_np[y:y2, x:x2]
            if dsm_np is not None:
                patch_dsm = dsm_np[y:y2, x:x2]
                pred = predict_patch_dsm(model, patch_rgb, patch_dsm)
            else:
                pred = predict_patch_rgb(model, patch_rgb)

            # Interior crop to avoid resize edge artifacts
            m = 16
            im, jm = min(m, ph//4), min(m, pw//4)
            pred_sum[y+im:y2-im, x+jm:x2-jm] += pred[im:ph-im, jm:pw-jm]
            count[y+im:y2-im, x+jm:x2-jm] += 1.0

    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile, img_dir, gt_dir, dsm_dir, img_suf, gt_suf, dsm_stem_fn, use_dsm):
    ip = f'{img_dir}/{tile}{img_suf}'
    gp = f'{gt_dir}/{tile}{gt_suf}'
    img = np.array(Image.open(ip).convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))

    dsm_np = None
    if use_dsm:
        dp = f'{dsm_dir}/dsm_{dsm_stem_fn(tile)}'
        if os.path.exists(dp):
            dsm_np = np.array(Image.open(dp)).astype(np.float32)
            dsm_np = (dsm_np - dsm_np.min()) / max(dsm_np.max() - dsm_np.min(), 1e-8)

    print(f"  {tile} ({img.shape[1]}x{img.shape[0]}) inferring...", end=' ', flush=True)
    pred = sliding_256(img.shape[0], img.shape[1], model, img, dsm_np)

    mask = gt != 255
    oa = (pred[mask] == gt[mask]).sum() / mask.sum() * 100
    total = mask.sum()
    ious = {}
    oas = {}
    for c in range(5):
        pc, lc = pred == c, gt == c
        inter = (pc & lc).sum()
        union = (pc | lc).sum()
        ious[CLASS_NAMES[c]] = (inter / union * 100) if union > 0 else 0.0
        tn = ((~pc) & (~lc) & mask).sum()
        oas[CLASS_NAMES[c]] = (inter + tn) / max(total, 1) * 100
    miou = np.mean(list(ious.values()))
    moa = np.mean(list(oas.values()))
    print(f"OA={oa:.1f}% mIoU={miou:.1f}% | per-cls OA={moa:.1f}%")
    return oa, miou, ious, oas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='rgb', choices=['rgb', 'dsm'])
    ap.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    args = ap.parse_args()

    ckpts = {
        ('rgb', 'vaihingen'): '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049/best_model.pt',
        ('dsm', 'vaihingen'): '/root/Mynet/autodl-tmp/runs/plan3_dual_20260502_160434/best_model.pt',
        ('rgb', 'potsdam'):  '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_185038/best_model.pt',
    }

    ckpt_path = ckpts.get((args.model, args.dataset))
    if not ckpt_path:
        print(f"No checkpoint for {args.model}/{args.dataset}"); return

    print(f"=== {args.dataset.upper()} | Model: {args.model.upper()} | 256² sliding ===")
    dropout = 0.0 if args.dataset == 'potsdam' else 0.1  # Potsdam trained without dropout
    if args.model == 'dsm':
        model, ckpt = load_dsm_model(ckpt_path)
    else:
        model, ckpt = load_rgb_model(ckpt_path, dropout=dropout)
    print(f"  Loaded epoch {ckpt['epoch']}, crop mIoU={ckpt['best_v']:.1f}%")

    if args.dataset == 'vaihingen':
        tiles = VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
        img_suf, gt_suf = '.tif', '.tif'
        dsm_fn = lambda t: f"09cm_matching_area{t.replace('top_mosaic_09cm_area', '')}.tif"
    else:
        tiles = POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'
        dsm_fn = lambda t: f"potsdam_{t.replace('top_potsdam_', '')}.tif"

    results = []
    for tile in tiles:
        oa, miou, ious, oas = evaluate_tile(
            model, tile, img_dir, gt_dir, dsm_dir, img_suf, gt_suf, dsm_fn,
            use_dsm=(args.model == 'dsm'))
        results.append({'tile': tile, 'oa': oa, 'miou': miou, **ious, **{f'{k}_oa': v for k, v in oas.items()}})

    # Summary
    avg_oa = np.mean([r['oa'] for r in results])
    avg_miou = np.mean([r['miou'] for r in results])
    avg_per_class = {c: np.mean([r[c] for r in results]) for c in CLASS_NAMES}
    avg_per_class_oa = {c: np.mean([r[f'{c}_oa'] for r in results]) for c in CLASS_NAMES}

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.dataset.upper()} | {args.model.upper()} | 256² protocol")
    print(f"  OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
    print(f"  Per-class IoU: " + ", ".join(f"{c}={avg_per_class[c]:.1f}" for c in CLASS_NAMES))
    print(f"  Per-class OA : " + ", ".join(f"{c}={avg_per_class_oa[c]:.1f}" for c in CLASS_NAMES))
    print(f"{'='*60}")

    out_dir = os.path.dirname(ckpt_path)
    with open(f'{out_dir}/eval_256_{args.dataset}_{args.model}.json', 'w') as f:
        json.dump({'avg_oa': avg_oa, 'avg_miou': avg_miou,
                   'per_class_iou': avg_per_class,
                   'per_class_oa': avg_per_class_oa,
                   'tiles': results}, f, indent=2)

    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
