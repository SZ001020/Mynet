#!/usr/bin/env python3
"""生成周报推理图：逐模型评估 + 全图对比 + 细节放大 + 逐类柱状图。
每个模型独立加载/推理/释放，预测存为 .npy，最后统一生成图。
用法: python utils/gen_weekly_vis.py --output weeklyReport/2026-05-24/
"""
import os, sys, argparse, json, gc
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = '/root/Mynet'
sys.path.insert(0, BASE)
sys.path.insert(0, f'{BASE}/SegEarth-OV-3-main')
sys.path.insert(0, f'{BASE}/RS-SAM-p3b')

PALETTE = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]]
CLASS_NAMES = ['road','building','grass','tree','car','clutter']
NUM_CLASSES = 5
IGNORE_INDEX = 255
DETAIL_SIZE = 200


def rgb_to_class(gt_rgb):
    gt = np.zeros(gt_rgb.shape[:2], dtype=np.int32)
    gt[(gt_rgb == [255,255,255]).all(2)] = 0
    gt[(gt_rgb == [0,0,255]).all(2)]   = 1
    gt[(gt_rgb == [0,255,255]).all(2)] = 2
    gt[(gt_rgb == [0,255,0]).all(2)]   = 3
    gt[(gt_rgb == [255,255,0]).all(2)] = 4
    gt[(gt_rgb == [255,0,0]).all(2)]   = 5
    return gt


def pred_to_color(pred, palette):
    h, w = pred.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        c[pred == i] = color
    return c


def sliding_inference(model, img, dsm, use_dsm=True, resolution=1008):
    """256² sliding window with soft-logit accumulation (per-patch argmax)."""
    H, W = img.shape[:2]
    ps = np.zeros((H, W), dtype=np.float64)
    ct = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - 128, 128):
        for x in range(0, W - 128, 128):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue
            patch = torch.from_numpy(img[y:y2, x:x2]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            patch = F.interpolate(patch, (resolution, resolution), mode='bilinear', align_corners=False)

            with torch.no_grad():
                if use_dsm and dsm is not None:
                    pd = torch.from_numpy(dsm[y:y2, x:x2]).float().unsqueeze(0)
                    pd = F.interpolate(pd.unsqueeze(1), (resolution, resolution),
                                       mode='bilinear', align_corners=False)
                    logits = model(patch.cuda(), pd.squeeze(1).cuda())
                else:
                    logits = model(patch.cuda())
                logits = F.interpolate(logits, (ph, pw), mode='bilinear', align_corners=False)
                p = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)

            im, jm = min(16, ph // 4), min(16, pw // 4)
            ps[y + im:y2 - im, x + jm:x2 - jm] += p[im:ph - im, jm:pw - jm]
            ct[y + im:y2 - im, x + jm:x2 - jm] += 1.0

    ct[ct == 0] = 1.0
    return np.round(ps / ct).astype(np.int64)


def get_tile_data(tile_id):
    ip = f'/root/autodl-tmp/dataset/Vaihingen/top/top_mosaic_09cm_area{tile_id}.tif'
    gp = f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area{tile_id}.tif'
    dp = f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{tile_id}.tif'

    rgb = np.array(Image.open(ip).convert('RGB'))
    gt_raw = np.array(Image.open(gp).convert('RGB'))
    gt = rgb_to_class(gt_raw)
    dsm = np.array(Image.open(dp)).astype(np.float32)
    dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    return rgb, gt, dsm


def find_detail_regions(gt, H, W):
    """Find 4 diverse detail regions."""
    candidates = []
    for y in range(0, H - DETAIL_SIZE, DETAIL_SIZE // 2):
        for x in range(0, W - DETAIL_SIZE, DETAIL_SIZE // 2):
            crop_gt = gt[y:y + DETAIL_SIZE, x:x + DETAIL_SIZE]
            crop_mask = crop_gt != IGNORE_INDEX
            if crop_mask.sum() < (DETAIL_SIZE * DETAIL_SIZE * 0.1):
                continue
            classes_present = set(np.unique(crop_gt[crop_mask]))
            classes_present.discard(IGNORE_INDEX)
            if len(classes_present) < 2:
                continue
            n_building = (crop_gt == 1).sum()
            candidates.append({'y': y, 'x': x, 'classes': classes_present,
                               'n_classes': len(classes_present), 'n_building': n_building})

    if len(candidates) < 4:
        return [(H//4, W//4), (H//4, 3*W//4 - DETAIL_SIZE),
                (3*H//4 - DETAIL_SIZE, W//4), (3*H//4 - DETAIL_SIZE, 3*W//4 - DETAIL_SIZE)]

    selected = []
    building_candidates = [c for c in candidates if 1 in c['classes']]
    if building_candidates:
        best = max(building_candidates, key=lambda c: c['n_building'])
        selected.append((best['y'], best['x']))
        candidates = [c for c in candidates if c != best]

    veg_candidates = [c for c in candidates if {2, 3} & c['classes']]
    if veg_candidates and len(selected) < 4:
        best = max(veg_candidates, key=lambda c: c['n_classes'])
        selected.append((best['y'], best['x']))
        candidates = [c for c in candidates if c != best]

    road_candidates = [c for c in candidates if 0 in c['classes']]
    if road_candidates and len(selected) < 4:
        best = max(road_candidates, key=lambda c: c['n_classes'])
        selected.append((best['y'], best['x']))
        candidates = [c for c in candidates if c != best]

    while len(selected) < 4 and candidates:
        best = max(candidates, key=lambda c: c['n_classes'])
        selected.append((best['y'], best['x']))
        candidates.remove(best)

    return selected[:4]


def compute_metrics(pred, gt):
    inter, union, correct, total_gt = {}, {}, {}, {}
    mask = gt != IGNORE_INDEX
    total_correct = (pred[mask] == gt[mask]).sum()
    total_pixels = mask.sum()

    for i, c in enumerate(CLASS_NAMES[:NUM_CLASSES]):
        pc = (pred == i); lc = (gt == i)
        inter[c] = (pc & lc).sum(); union[c] = (pc | lc).sum()
        correct[c] = inter[c]; total_gt[c] = lc.sum()

    return {
        'oa': total_correct / max(total_pixels, 1) * 100,
        'miou': np.mean([inter[c] / max(union[c], 1) * 100 for c in CLASS_NAMES[:NUM_CLASSES]]),
        'per_class_iou': {c: inter[c] / max(union[c], 1) * 100 for c in CLASS_NAMES[:NUM_CLASSES]},
        'per_class_recall': {c: correct[c] / max(total_gt[c], 1) * 100 for c in CLASS_NAMES[:NUM_CLASSES]},
    }


def load_sam3():
    _prev = os.getcwd()
    os.chdir(f'{BASE}/SegEarth-OV-3-main')
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(
        bpe_path=f'{BASE}/SegEarth-OV-3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{BASE}/SegEarth-OV-3-main/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    return m.cuda()


# ── Visualization functions (no model dependency) ──

def generate_panorama(rgb, gt, preds_dict, out_path, title):
    n_preds = len(preds_dict)
    fig, axes = plt.subplots(1, 2 + n_preds, figsize=(5 * (2 + n_preds), 5))
    if 2 + n_preds == 1:
        axes = [axes]

    max_disp = 600
    ratio = max_disp / max(rgb.shape[:2])
    def ds(img):
        interp = Image.LANCZOS if img.ndim == 3 else Image.NEAREST
        return np.array(Image.fromarray(img).resize(
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interp))

    gt_color = pred_to_color(np.clip(gt, 0, 5), PALETTE)

    axes[0].imshow(ds(rgb)); axes[0].set_title('RGB', fontweight='bold', fontsize=10); axes[0].axis('off')
    axes[1].imshow(ds(gt_color)); axes[1].set_title('Ground Truth', fontweight='bold', fontsize=10); axes[1].axis('off')

    for i, (label, pred) in enumerate(preds_dict.items()):
        pred_color = pred_to_color(pred, PALETTE[:5] + PALETTE[5:])
        axes[2 + i].imshow(ds(pred_color))
        axes[2 + i].set_title(label, fontweight='bold', fontsize=10)
        axes[2 + i].axis('off')

    patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
               for c, n in zip(PALETTE[:NUM_CLASSES], CLASS_NAMES[:NUM_CLASSES])]
    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES, fontsize=8, frameon=False)
    fig.suptitle(title, fontweight='bold', y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Panorama saved: {out_path}")


def generate_details(rgb, gt, preds_dict, regions, out_path, title_prefix):
    n_preds = len(preds_dict)
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 2 + n_preds,
                             figsize=(3 * (2 + n_preds), 3 * n_regions))
    if n_regions == 1:
        axes = axes.reshape(1, -1)

    col_labels = ['RGB', 'GT'] + list(preds_dict.keys())
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontweight='bold', fontsize=9)

    def crop(img, ry, rx):
        return img[ry:ry + DETAIL_SIZE, rx:rx + DETAIL_SIZE]

    gt_color = pred_to_color(np.clip(gt, 0, 5), PALETTE)
    for i, (ry, rx) in enumerate(regions):
        axes[i, 0].imshow(crop(rgb, ry, rx)); axes[i, 0].axis('off')
        axes[i, 1].imshow(crop(gt_color, ry, rx)); axes[i, 1].axis('off')
        for k, (label, pred) in enumerate(preds_dict.items()):
            pred_color = pred_to_color(pred, PALETTE[:5] + PALETTE[5:])
            axes[i, 2 + k].imshow(crop(pred_color, ry, rx))
            axes[i, 2 + k].axis('off')

    patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
               for c, n in zip(PALETTE[:NUM_CLASSES], CLASS_NAMES[:NUM_CLASSES])]
    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES, fontsize=8, frameon=False)
    fig.suptitle(f'{title_prefix} — Detail Regions', fontweight='bold', y=0.98, fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Details saved: {out_path}")


def generate_perclass_bars(metrics_dict, out_path):
    n_models = len(metrics_dict)
    classes = CLASS_NAMES[:NUM_CLASSES]
    x = np.arange(len(classes))
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, m) in enumerate(metrics_dict.items()):
        values = [m['per_class_iou'][c] for c in classes]
        bars = ax.bar(x + i * width, values, width, label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', fontsize=7)

    ax.set_ylabel('IoU (%)', fontsize=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(fontsize=9)
    all_vals = [m['per_class_iou'][c] for m in metrics_dict.values() for c in classes]
    ax.set_ylim(0, max(all_vals) + 15)
    fig.suptitle('Per-Class IoU Comparison', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bars saved: {out_path}")


# ── Main: one model at a time ──

def run_one_model(label, loader, ckpt_path, tiles, cache_dir):
    """Load model, run inference on all tiles, save predictions, return metrics."""
    print(f"\n{'='*60}")
    print(f"Loading: {label}")
    print(f"  Checkpoint: {ckpt_path}")

    # Import model class from correct source
    if loader == 'plan4_full':
        sys.path.insert(0, f'{BASE}/RS-SAM3-p4')
        from train_full import SAM3FullTrain
    elif loader == 'plan5':
        sys.path.insert(0, f'{BASE}/RS-SAM3-p5')
        from train import VPT_MFNetDecoder
    else:
        raise ValueError(f"Unknown loader: {loader}")

    sam3 = load_sam3()
    if loader == 'plan4_full':
        model = SAM3FullTrain(sam3, num_classes=5, use_dsm=True, dropout=0.1).cuda()
    else:
        model = VPT_MFNetDecoder(sam3, adapter_bottleneck=32, num_classes=5, use_dsm=True, dropout=0.1).cuda()

    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}, best_v={ckpt.get('best_v', 0):.1f}%")

    # Accumulate metrics across tiles
    all_inter = {c: 0.0 for c in CLASS_NAMES[:NUM_CLASSES]}
    all_union = {c: 0.0 for c in CLASS_NAMES[:NUM_CLASSES]}
    all_correct = {c: 0.0 for c in CLASS_NAMES[:NUM_CLASSES]}
    all_gt = {c: 0.0 for c in CLASS_NAMES[:NUM_CLASSES]}
    total_correct = 0.0
    total_pixels = 0.0

    for tile_id in tiles:
        print(f"  Tile {tile_id}...", end=' ', flush=True)
        rgb, gt, dsm = get_tile_data(tile_id)
        pred = sliding_inference(model, rgb, dsm, use_dsm=True)
        np.save(os.path.join(cache_dir, f'pred_{loader}_{tile_id}.npy'), pred)
        print(f"saved ({pred.shape})", flush=True)

        mask = gt != IGNORE_INDEX
        total_correct += (pred[mask] == gt[mask]).sum()
        total_pixels += mask.sum()
        for i, c in enumerate(CLASS_NAMES[:NUM_CLASSES]):
            pc = (pred == i); lc = (gt == i)
            all_inter[c] += (pc & lc).sum(); all_union[c] += (pc | lc).sum()
            all_correct[c] += (pc & lc).sum(); all_gt[c] += lc.sum()

    oa = total_correct / max(total_pixels, 1) * 100
    pc_iou = {c: all_inter[c] / max(all_union[c], 1) * 100 for c in CLASS_NAMES[:NUM_CLASSES]}
    pc_recall = {c: all_correct[c] / max(all_gt[c], 1) * 100 for c in CLASS_NAMES[:NUM_CLASSES]}
    miou = np.mean(list(pc_iou.values()))
    mrecall = np.mean(list(pc_recall.values()))

    metrics = {'oa': oa, 'miou': miou, 'mrecall': mrecall,
               'per_class_iou': pc_iou, 'per_class_recall': pc_recall}
    print(f"  OA={oa:.2f}%  mIoU={miou:.2f}%  mRecall={mrecall:.2f}%")

    del model, sam3
    gc.collect(); torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=f'{BASE}/weeklyReport/2026-05-24')
    parser.add_argument('--tile', default='21', help='Main tile for panorama + details')
    parser.add_argument('--tiles', nargs='*', default=['5', '15', '21', '30'])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cache_dir = os.path.join(args.output, '.pred_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # ── Model configs ──
    models_cfg = [
        {
            'label': 'Full Training (456M)',
            'loader': 'plan4_full',
            'ckpt': '/root/autodl-tmp/runs/plan4_full_vaihingen_20260507_175600/best_model.pt',
        },
        {
            'label': 'Frozen + Boundary Loss (5M)',
            'loader': 'plan5',
            'ckpt': '/root/autodl-tmp/runs/plan5_20260508_153225/best_model.pt',
        },
    ]

    # ── Run each model independently ──
    all_metrics = {}
    for cfg in models_cfg:
        if os.path.exists(cfg['ckpt']):
            all_metrics[cfg['label']] = run_one_model(
                cfg['label'], cfg['loader'], cfg['ckpt'], args.tiles, cache_dir)

    # Save metrics
    json_path = os.path.join(args.output, 'eval_metrics.json')
    json.dump(all_metrics, open(json_path, 'w'), indent=2)
    print(f"\nMetrics saved: {json_path}")

    # ── Generate visualizations ──
    print(f"\n{'='*60}")
    print(f"Generating visualizations (tile {args.tile})...")
    main_rgb, main_gt, _ = get_tile_data(args.tile)

    # Load predictions for main tile
    preds_dict = {}
    for cfg in models_cfg:
        pred_path = os.path.join(cache_dir, f'pred_{cfg["loader"]}_{args.tile}.npy')
        if os.path.exists(pred_path):
            preds_dict[cfg['label']] = np.load(pred_path)
            print(f"  Loaded {cfg['label']}: {preds_dict[cfg['label']].shape}")

    if not preds_dict:
        print("No predictions found!")
        return

    # 1. Panorama
    panorama_path = os.path.join(args.output, 'full_comparison.png')
    generate_panorama(main_rgb, main_gt, preds_dict, panorama_path,
                      f'Vaihingen Area {args.tile} — Model Comparison')

    # 2. Detail regions
    first_pred = list(preds_dict.values())[0]
    H, W = main_gt.shape
    regions = find_detail_regions(main_gt, H, W)
    print(f"  Detail regions: {regions}")
    detail_path = os.path.join(args.output, 'detail_comparison.png')
    generate_details(main_rgb, main_gt, preds_dict, regions, detail_path,
                     f'Vaihingen Area {args.tile}')

    # 3. Per-class bars
    bars_path = os.path.join(args.output, 'perclass_bars.png')
    generate_perclass_bars(all_metrics, bars_path)

    # Cleanup cache
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    print("\nAll done!")


if __name__ == '__main__':
    main()
