#!/usr/bin/env python3
"""
DSM Hillshade 快速验证：对比同一模型在 RGB vs RGB+DSM(hillshade) 下的推理差异
"""

import os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/RS-SAM3-p3')
from adapter_unet import AdapterSAM3UNet
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL

PALETTE = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]
CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']

def pred_to_color(pred):
    c = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        c[pred == i] = color
    c[pred == 255] = [255,0,0]
    return c

def compute_hillshade_slope(dsm, resolution=0.09):
    """Compute hillshade + slope from DSM (meters)."""
    from scipy import ndimage
    # Gradient
    dy, dx = np.gradient(dsm.astype(np.float64))
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / resolution)
    slope = slope_rad / (np.pi / 2)  # normalize to [0, 1]

    # Hillshade (sun at azimuth 315°, altitude 45°)
    az_rad = np.deg2rad(315)
    alt_rad = np.deg2rad(45)
    dz = 1.0
    hillshade = (np.cos(alt_rad) * dz +
                 np.sin(alt_rad) * dz *
                 (np.cos(az_rad) * dx + np.sin(az_rad) * dy) /
                 np.sqrt(dx**2 + dy**2 + dz**2))
    hillshade = np.clip(hillshade, 0, 1)
    # Handle flat areas (dx=dy=0)
    hillshade = np.nan_to_num(hillshade, nan=0.5)
    return hillshade, slope

def dsm_to_rgb_overlay(rgb, dsm, alpha=0.6):
    """Overlay DSM hillshade+slope onto RGB as 3-channel enhanced image."""
    hillshade, slope = compute_hillshade_slope(dsm)
    # Normalize RGB to [0, 1]
    rgb_f = rgb.astype(np.float32) / 255.0
    # Blend: R=RGB_R, G=RGB_G enhanced with hillshade, B=RGB_B enhanced with slope
    # But keep original RGB mostly, overlay DSM info
    enhanced = rgb_f.copy()
    # Hillshade modulates brightness (like topographic relief)
    enhanced = enhanced * (0.7 + 0.3 * hillshade[:, :, np.newaxis])
    # Slope adds texture (steeper = brighter in blue channel)
    enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] + 0.3 * slope, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def load_model(ckpt_path):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    model = AdapterSAM3UNet(sam3, adapter_bottleneck=32, num_classes=5,
                            dropout=0.1).cuda()
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model

@torch.no_grad()
def predict_patch(model, img_patch):
    """Predict on a single 512² patch."""
    img_t = torch.from_numpy(img_patch).permute(2,0,1).float().unsqueeze(0)/255.0
    logits = model(img_t.cuda())
    logits = F.interpolate(logits, (512,512), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()

def run_experiment(dataset='vaihingen'):
    ckpt = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049/best_model.pt'
    out_dir = f'/root/Mynet/autodl-tmp/runs/dsm_quick_test_{dataset}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model...")
    model = load_model(ckpt)

    if dataset == 'vaihingen':
        tiles = VAIHINGEN_VAL[:2]  # first 2 validation tiles
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
        img_suf, gt_suf = '.tif', '.tif'
        dsm_pattern = 'dsm_09cm_matching_area{}.tif'
        tile_stems = [t.replace('top_mosaic_09cm_area', '') for t in tiles]
    else:
        tiles = POTSDAM_VAL[:2]
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'
        dsm_pattern = 'dsm_potsdam_{}.tif'
        tile_stems = [t.replace('top_potsdam_', '') for t in tiles]

    results = {'rgb': [], 'dsm': []}

    for k, (tile, stem) in enumerate(zip(tiles, tile_stems)):
        ip = f'{img_dir}/{tile}{img_suf}'
        gp = f'{gt_dir}/{tile}{gt_suf}'
        dp = f'{dsm_dir}/{dsm_pattern.format(stem)}'

        if not os.path.exists(dp):
            print(f"  Skip {tile}: no DSM at {dp}")
            continue

        img_full = np.array(Image.open(ip).convert('RGB'))
        gt_rgb = np.array(Image.open(gp).convert('RGB'))
        gt_idx = _rgb_to_class(gt_rgb)
        dsm = np.array(Image.open(dp))

        # Ensure DSM matches image size
        if dsm.shape[:2] != img_full.shape[:2]:
            dsm = np.array(Image.fromarray(dsm).resize(
                (img_full.shape[1], img_full.shape[0]), Image.BILINEAR))

        # Generate DSM-enhanced RGB
        img_enhanced = dsm_to_rgb_overlay(img_full, dsm)

        # Take 4 center crops for robust comparison
        h, w = img_full.shape[:2]
        crops = [(h//2-256, h//2+256, w//2-256, w//2+256),
                 (256, 768, 256, 768),
                 (h-768, h-256, w-768, w-256)]

        for ci, (y1,y2,x1,x2) in enumerate(crops):
            if y1 < 0 or y2 > h or x1 < 0 or x2 > w: continue

            gt_crop = gt_idx[y1:y2, x1:x2]
            mask = gt_crop != 255
            if mask.sum() == 0: continue

            # RGB prediction
            rgb_crop = img_full[y1:y2, x1:x2]
            pred_rgb = predict_patch(model, rgb_crop)
            oa_rgb = (pred_rgb[mask] == gt_crop[mask]).sum() / mask.sum() * 100

            # DSM-enhanced prediction
            dsm_crop = img_enhanced[y1:y2, x1:x2]
            pred_dsm = predict_patch(model, dsm_crop)
            oa_dsm = (pred_dsm[mask] == gt_crop[mask]).sum() / mask.sum() * 100

            results['rgb'].append(oa_rgb)
            results['dsm'].append(oa_dsm)

            # Save first crop visual comparison
            if ci == 0 and k < 2:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes[0,0].imshow(rgb_crop); axes[0,0].set_title('RGB Input')
                axes[0,1].imshow(pred_to_color(gt_crop)); axes[0,1].set_title('Ground Truth')
                axes[0,2].imshow(pred_to_color(pred_rgb))
                axes[0,2].set_title(f'RGB pred (OA={oa_rgb:.1f}%)')
                axes[1,0].imshow(dsm_crop); axes[1,0].set_title('RGB+DSM Input')
                axes[1,1].imshow(pred_to_color(gt_crop)); axes[1,1].set_title('Ground Truth')
                axes[1,2].imshow(pred_to_color(pred_dsm))
                axes[1,2].set_title(f'RGB+DSM pred (OA={oa_dsm:.1f}%)')
                for ax in axes.flat: ax.axis('off')
                fig.savefig(f'{out_dir}/compare_{tile}_{ci}.png', dpi=120, bbox_inches='tight')
                plt.close(fig)

            print(f"  {tile} crop{ci}: RGB={oa_rgb:.1f}%  RGB+DSM={oa_dsm:.1f}%  "
                  f"Δ={oa_dsm-oa_rgb:+.1f}%")

    avg_rgb = np.mean(results['rgb'])
    avg_dsm = np.mean(results['dsm'])
    print(f"\n{'='*50}")
    print(f"AVERAGE: RGB={avg_rgb:.1f}%  RGB+DSM={avg_dsm:.1f}%  "
          f"Δ={avg_dsm-avg_rgb:+.1f}% ({len(results['rgb'])} crops)")
    print(f"{'='*50}")

    del model; torch.cuda.empty_cache()
    return avg_rgb, avg_dsm

if __name__ == '__main__':
    for ds in ['vaihingen', 'potsdam']:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*50}")
        try:
            run_experiment(ds)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
