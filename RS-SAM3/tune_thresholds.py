#!/usr/bin/env python3
"""
Plan2 Phase 1b: Per-Class Threshold Calibration

在 Vaihingen 4 张验证图上做 per-class 阈值扫描，
找到最大化 Dice/IoU 的最优 semantic head 激活百分位，
然后应用到全量评估。

方式: 扫描 top-K% (K ∈ [3, 10, 15, 20, 25, 30, 35, 40, 50])
"""

import sys, os, json, csv
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/root/Mynet/RS-SAM3')
os.chdir('/root/Mynet/SegEarth-OV-3-main')

from sam3_model import SAM3Model
from dataset_rs import load_rs_samples, CLASSES, TEXT_PROMPTS
from metrics import compute_all_metrics, resize_mask

# Calibration tiles (held-out from training, used as validation)
VAL_TILES_VAHINGEN = [
    'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32',
    'top_mosaic_09cm_area34', 'top_mosaic_09cm_area37',
]
VAL_TILES_POTSDAM = [
    'top_potsdam_6_7', 'top_potsdam_6_8', 'top_potsdam_6_9',
    'top_potsdam_7_7', 'top_potsdam_7_8', 'top_potsdam_7_9',
]

# Threshold percentiles to sweep
K_VALUES = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]


def predict_with_threshold(sam3, inference_state, text_prompt, top_k_pct):
    """Per-class threshold version of predict_binary."""
    sam3.processor.reset_all_prompts(inference_state)
    state = sam3.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    h, w = state['original_height'], state['original_width']
    combined = torch.zeros((h, w), dtype=torch.bool, device=sam3.device)

    # Instance masks (confidence-independent, always included)
    masks = state.get('masks')
    if masks is not None and len(masks) > 0:
        for m in masks[:10]:
            combined = combined | m.bool()

    # Semantic head with variable threshold
    sem = state.get('semantic_mask_logits')
    if sem is not None:
        from torch.nn.functional import interpolate
        sem_prob = sem.squeeze().sigmoid()
        if sem_prob.dim() > 2:
            sem_prob = sem_prob.squeeze()
        k = max(int(sem_prob.numel() * top_k_pct / 100.0), 50)
        threshold = sem_prob.flatten().topk(k).values[-1].item()
        sem_mask = sem_prob > threshold
        if sem_mask.shape != (h, w):
            sem_mask = interpolate(sem_mask.float().unsqueeze(0).unsqueeze(0),
                                   (h, w), mode='nearest').squeeze() > 0.5
        combined = combined | sem_mask

    return np.squeeze(combined.cpu().numpy().astype(np.uint8))


def calibrate(sam3, dataset_name, val_tiles):
    """Sweep K values per class on validation tiles, return best K per class."""
    all_samples = list(load_rs_samples(dataset_name))
    # Filter to val tiles only
    val_samples = [s for s in all_samples if any(vt in s.tile_name for vt in val_tiles)]
    print(f"  Calibration: {len(val_samples)} samples ({len(val_tiles)} tiles)")

    # Group by class
    by_class = defaultdict(list)
    for s in val_samples:
        by_class[s.class_name].append(s)

    best_k = {}
    calibration_data = []

    for cls_name, cls_label in CLASSES:
        samples = by_class[cls_name]
        print(f"\n  {cls_name}: {len(samples)} samples")
        best_iou = 0.0
        best_k_for_cls = K_VALUES[3]  # default 15%

        for k in K_VALUES:
            metrics_list = []
            for sample in samples:
                state = sam3.encode_image(sample.image)
                pred = predict_with_threshold(sam3, state, TEXT_PROMPTS[cls_name], k)
                pred = np.squeeze(pred)
                if pred.shape[:2] != sample.gt_mask.shape[:2]:
                    pred = resize_mask(pred, sample.gt_mask.shape[:2])
                m = compute_all_metrics(pred, sample.gt_mask)
                metrics_list.append(m)

            avg_iou = np.mean([m['iou'] for m in metrics_list])
            avg_dice = np.mean([m['dice'] for m in metrics_list])
            calibration_data.append({
                'dataset': dataset_name, 'class': cls_name, 'k_pct': k,
                'iou': round(avg_iou, 4), 'dice': round(avg_dice, 4),
            })

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_k_for_cls = k

        best_k[cls_name] = best_k_for_cls
        print(f"    Best K={best_k_for_cls}% (IoU={best_iou:.3f})")

    return best_k, calibration_data


def evaluate_with_per_class_thresholds(sam3, dataset_name, best_k, max_samples=None):
    """Full evaluation using per-class optimal thresholds."""
    samples = list(load_rs_samples(dataset_name, max_samples))
    all_metrics = defaultdict(list)

    for sample in tqdm(samples, desc=f'{dataset_name} (tuned)'):
        k = best_k.get(sample.class_name, 15)
        state = sam3.encode_image(sample.image)
        pred = predict_with_threshold(sam3, state, TEXT_PROMPTS[sample.class_name], k)
        pred = np.squeeze(pred)
        if pred.shape[:2] != sample.gt_mask.shape[:2]:
            pred = resize_mask(pred, sample.gt_mask.shape[:2])
        m = compute_all_metrics(pred, sample.gt_mask)
        m['class'] = sample.class_name
        m['tile'] = sample.tile_name
        m['dataset'] = dataset_name
        all_metrics[sample.class_name].append(m)

    results = {}
    for cls_name in sorted(all_metrics.keys()):
        ms = all_metrics[cls_name]
        agg = {}
        for key in ['dice', 'iou', 'precision', 'recall']:
            vals = [m[key] for m in ms]
            agg[f'{key}_mean'] = np.mean(vals)
            agg[f'{key}_std'] = np.std(vals)
        results[cls_name] = agg
        print(f"  {cls_name:>10}: Dice={agg['dice_mean']:.3f}  IoU={agg['iou_mean']:.3f}  "
              f"Prec={agg['precision_mean']:.3f}  Rec={agg['recall_mean']:.3f}  (K={best_k[cls_name]}%)")

    return results


def main():
    sam3 = SAM3Model(confidence_threshold=0.1)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'/root/Mynet/autodl-tmp/runs/plan2_phase1b_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # === Step 1: Calibrate per dataset ===
    print("=" * 60)
    print("STEP 1: Per-Class Threshold Calibration")
    print("=" * 60)

    all_calib = []
    best_thresholds = {}

    for ds_name, val_tiles in [('vaihingen', VAL_TILES_VAHINGEN), ('potsdam', VAL_TILES_POTSDAM)]:
        print(f"\nCalibrating {ds_name}...")
        best_k, calib_data = calibrate(sam3, ds_name, val_tiles)
        best_thresholds[ds_name] = best_k
        all_calib.extend(calib_data)

    # Save calibration
    with open(os.path.join(output_dir, 'calibration.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','class','k_pct','iou','dice'])
        writer.writeheader()
        for row in all_calib:
            writer.writerow(row)

    # === Step 2: Full evaluation with tuned thresholds ===
    print(f"\n{'='*60}")
    print("STEP 2: Full Evaluation with Tuned Thresholds")
    print(f"{'='*60}")

    all_results = {}
    for ds_name in ['vaihingen', 'potsdam']:
        print(f"\n{ds_name}:")
        results = evaluate_with_per_class_thresholds(sam3, ds_name, best_thresholds[ds_name])
        all_results[ds_name] = results

    # === Step 3: Compare with baseline ===
    print(f"\n{'='*60}")
    print("COMPARISON: Baseline vs Tuned")
    print(f"{'='*60}")

    # Baseline numbers from Phase 1
    baseline = {
        'vaihingen': {
            'building': {'dice': 0.777, 'iou': 0.648, 'prec': 0.964, 'rec': 0.667},
            'car':      {'dice': 0.662, 'iou': 0.501, 'prec': 0.601, 'rec': 0.756},
            'grass':    {'dice': 0.399, 'iou': 0.255, 'prec': 0.740, 'rec': 0.309},
            'road':     {'dice': 0.592, 'iou': 0.423, 'prec': 0.957, 'rec': 0.433},
            'tree':     {'dice': 0.647, 'iou': 0.493, 'prec': 0.927, 'rec': 0.523},
        },
        'potsdam': {
            'building': {'dice': 0.752, 'iou': 0.614, 'prec': 0.950, 'rec': 0.640},
            'car':      {'dice': 0.744, 'iou': 0.592, 'prec': 0.704, 'rec': 0.793},
            'grass':    {'dice': 0.489, 'iou': 0.329, 'prec': 0.755, 'rec': 0.379},
            'road':     {'dice': 0.581, 'iou': 0.412, 'prec': 0.871, 'rec': 0.444},
            'tree':     {'dice': 0.373, 'iou': 0.235, 'prec': 0.898, 'rec': 0.242},
        },
    }

    print(f"{'Dataset':<12} {'Class':<12} {'Base IoU':>10} {'Tuned IoU':>10} {'Δ':>8} {'Best K':>8}")
    print("-" * 62)

    summary_rows = []
    for ds_name in ['vaihingen', 'potsdam']:
        for cls_name, cls_label in CLASSES:
            base = baseline[ds_name][cls_name]
            tuned = all_results[ds_name][cls_name]
            delta = tuned['iou_mean'] - base['iou']
            k = best_thresholds[ds_name][cls_name]
            print(f"{ds_name:<12} {cls_name:<12} {base['iou']:>9.3f} {tuned['iou_mean']:>9.3f} {delta:>+7.3f} {k:>7}%")
            summary_rows.append({
                'dataset': ds_name, 'class': cls_name,
                'base_iou': base['iou'], 'tuned_iou': round(tuned['iou_mean'], 4),
                'delta': round(delta, 4), 'best_k': k,
                'base_dice': base['dice'], 'tuned_dice': round(tuned['dice_mean'], 4),
                'base_prec': base['prec'], 'tuned_prec': round(tuned['precision_mean'], 4),
                'base_rec': base['rec'], 'tuned_rec': round(tuned['recall_mean'], 4),
            })

    # Save
    with open(os.path.join(output_dir, 'tuned_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    json.dump(summary_rows, open(os.path.join(output_dir, 'results.json'), 'w'), indent=2)

    # Overall
    base_miou = np.mean([r['base_iou'] for r in summary_rows])
    tuned_miou = np.mean([r['tuned_iou'] for r in summary_rows])
    print(f"\n  Overall: Baseline mIoU={base_miou:.3f} → Tuned mIoU={tuned_miou:.3f} (Δ={tuned_miou-base_miou:+.3f})")

    sam3.cleanup()
    print(f"\nOutput: {output_dir}")


if __name__ == '__main__':
    main()
