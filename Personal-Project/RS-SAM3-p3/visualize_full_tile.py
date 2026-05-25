#!/usr/bin/env python3
"""
全图推理可视化：滑动窗口推理整张 ISPRS tile，生成 Input/GT/Pred 对比图
"""

import os, sys, numpy as np, torch, torch.nn.functional as F, argparse
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p3')
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL
from adapter_unet import AdapterSAM3UNet

PALETTE = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]
CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
IGNORE_COLOR = [255,0,0]


def pred_to_color(pred):
    c = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        c[pred == i] = color
    c[pred == 255] = IGNORE_COLOR
    return c


def compute_metrics(pred, gt):
    """Per-class IoU + OA for a full tile."""
    mask = gt != 255
    total = mask.sum()
    oa = (pred[mask] == gt[mask]).sum() / max(total, 1) * 100
    ious = {}
    oas = {}
    for c in range(5):
        pc, lc = pred == c, gt == c
        inter = (pc & lc).sum()
        union = (pc | lc).sum()
        ious[CLASS_NAMES[c]] = (inter / union * 100) if union > 0 else 0.0
        tn = ((~pc) & (~lc) & mask).sum()
        oas[CLASS_NAMES[c]] = (inter + tn) / max(total, 1) * 100
    mIoU = np.mean(list(ious.values()))
    return oa, mIoU, ious, oas


def sliding_inference(model, img_np, window=1008, stride=672, device='cuda'):
    """Sliding window inference with overlap blending."""
    h, w = img_np.shape[:2]
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    pred_sum = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2, x2 = min(y + window, h), min(x + window, w)
            ph, pw = y2 - y, x2 - x

            patch = img_t[:, y:y2, x:x2]
            if ph < window or pw < window:
                patch = F.pad(patch, (0, window - pw, 0, window - ph))

            with torch.no_grad():
                logits = model(patch.unsqueeze(0).to(device))
                logits = F.interpolate(logits, (window, window),
                                       mode='bilinear', align_corners=False)
                p = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)

            pred_sum[y:y2, x:x2] += p
            count[y:y2, x:x2] += 1.0

    pred = np.round(pred_sum / count).astype(np.int64)
    return pred


def load_model(ckpt_path, dropout=0.1, device='cuda'):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    model = AdapterSAM3UNet(sam3, adapter_bottleneck=32, num_classes=5,
                            dropout=dropout).to(device)
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt


def run_dataset(model, tiles, img_dir, gt_dir, img_suffix, gt_suffix, out_dir, tag):
    """Run full-tile inference on a list of tiles, save comparison images."""
    results = []
    for ti, tile in enumerate(tiles):
        ip = f'{img_dir}/{tile}{img_suffix}'
        gp = f'{gt_dir}/{tile}{gt_suffix}'
        if not os.path.exists(ip):
            print(f"  [{ti+1}/{len(tiles)}] {tile} — SKIP (no image)")
            continue

        print(f"  [{ti+1}/{len(tiles)}] {tile} loading...", end=' ', flush=True)
        img_np = np.array(Image.open(ip).convert('RGB'))
        gt_rgb = np.array(Image.open(gp).convert('RGB'))
        gt_idx = _rgb_to_class(gt_rgb)

        print(f"inferring ({img_np.shape[1]}x{img_np.shape[0]})...", end=' ', flush=True)
        pred = sliding_inference(model, img_np)

        oa, miou, ious, oas = compute_metrics(pred, gt_idx)
        print(f"OA={oa:.1f}% mIoU={miou:.1f}%")

        # Downscale for visualization (tiles are 2000-6000px, too large for matplotlib)
        max_dim = 2000
        h, w = img_np.shape[:2]
        scale = min(max_dim / h, max_dim / w)
        if scale < 1:
            dh, dw = int(h * scale), int(w * scale)
            img_viz = np.array(Image.fromarray(img_np).resize((dw, dh), Image.BILINEAR))
            gt_viz = np.array(Image.fromarray(pred_to_color(gt_idx)).resize((dw, dh), Image.NEAREST))
            pred_viz = np.array(Image.fromarray(pred_to_color(pred)).resize((dw, dh), Image.NEAREST))
        else:
            img_viz, gt_viz, pred_viz = img_np, pred_to_color(gt_idx), pred_to_color(pred)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        axes[0].imshow(img_viz)
        axes[0].set_title(f'Input\n{tile}', fontsize=12); axes[0].axis('off')
        axes[1].imshow(gt_viz)
        axes[1].set_title('Ground Truth', fontsize=12); axes[1].axis('off')
        axes[2].imshow(pred_viz)
        axes[2].set_title(f'Fine-tuned (OA={oa:.1f}%, mIoU={miou:.1f}%)', fontsize=12); axes[2].axis('off')

        legend = [plt.Rectangle((0,0),1,1, fc=np.array(c)/255) for c in PALETTE]
        fig.legend(legend, CLASS_NAMES, loc='lower center', ncol=5, fontsize=10, frameon=False)
        fig.savefig(f'{out_dir}/full_{tag}_{tile}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        results.append({'tile': tile, 'oa': float(oa), 'miou': float(miou),
                        'per_class_iou': {k: float(v) for k, v in ious.items()},
                        'per_class_oa': {k: float(v) for k, v in oas.items()}})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    ap.add_argument('--ckpt', default=None, help='Checkpoint path')
    ap.add_argument('--out', default=None, help='Output directory')
    ap.add_argument('--stride', type=int, default=672)
    args = ap.parse_args()

    if args.dataset == 'vaihingen':
        if args.ckpt is None:
            args.ckpt = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049/best_model.pt'
        if args.out is None:
            args.out = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049'
        tiles = VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        img_suf, gt_suf = '.tif', '.tif'
    else:
        if args.ckpt is None:
            args.ckpt = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_185038/best_model.pt'
        if args.out is None:
            args.out = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_185038'
        tiles = POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'

    print(f"=== {args.dataset.upper()} Full-Tile Inference ===")
    print(f"  Tiles: {len(tiles)}")
    print(f"  Stride: {args.stride} (window=1008)")

    # Old Potsdam checkpoint was trained without Dropout
    dropout = 0.0 if args.dataset == 'potsdam' else 0.1
    model, ckpt = load_model(args.ckpt, dropout=dropout)
    print(f"  Model: epoch {ckpt['epoch']}, mIoU={ckpt['best_v']:.1f}%")

    results = run_dataset(model, tiles, img_dir, gt_dir, img_suf, gt_suf,
                          args.out, args.dataset)

    # Summary
    avg_oa = np.mean([r['oa'] for r in results])
    avg_miou = np.mean([r['miou'] for r in results])
    avg_pc_iou = {c: float(np.mean([r['per_class_iou'][c] for r in results])) for c in CLASS_NAMES}
    avg_pc_oa = {c: float(np.mean([r['per_class_oa'][c] for r in results])) for c in CLASS_NAMES}
    print(f"\n=== Summary ===")
    print(f"  Avg OA: {avg_oa:.1f}%")
    print(f"  Avg mIoU: {avg_miou:.1f}%")
    print(f"  Per-class IoU: " + ", ".join(f"{c}={avg_pc_iou[c]:.1f}" for c in CLASS_NAMES))
    print(f"  Per-class OA : " + ", ".join(f"{c}={avg_pc_oa[c]:.1f}" for c in CLASS_NAMES))
    for r in results:
        print(f"  {r['tile']}: OA={r['oa']:.1f}%, mIoU={r['miou']:.1f}%, "
              + ', '.join(f"{k}={v:.1f}" for k, v in r['per_class_iou'].items()))

    # Save JSON
    import json
    with open(f'{args.out}/full_tile_results_{args.dataset}.json', 'w') as f:
        json.dump({'avg_oa': float(avg_oa), 'avg_miou': float(avg_miou),
                   'per_class_iou': avg_pc_iou, 'per_class_oa': avg_pc_oa,
                   'tiles': results}, f, indent=2)

    del model; torch.cuda.empty_cache()
    print(f"\nDone! → {args.out}/")

if __name__ == '__main__':
    main()
