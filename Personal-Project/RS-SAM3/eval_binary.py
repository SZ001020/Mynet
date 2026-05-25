#!/usr/bin/env python3
"""
Plan2 Phase 1: Per-Class Binary Classification Baseline

对 Vaihingen/Potsdam 的 5 个遥感类别独立做二分类评估。
使用 SAM3 text prompt ("road", "building", etc.) → binary mask → Dice/IoU.

输出格式:
  runs/plan2_phase1_{timestamp}/
  ├── experiment.log
  ├── all_results.csv
  ├── {dataset}_results.csv
  └── {dataset}_{tile}_{class}_pred.png  (per-class binary mask visualizations)
"""

import os, sys, json, time, gc, argparse, csv
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3')

from sam3_model import SAM3Model
from dataset_rs import load_rs_samples, CLASSES, TEXT_PROMPTS
from metrics import compute_all_metrics, resize_mask


def evaluate_dataset(sam3: SAM3Model, dataset_name: str, max_samples=None, output_dir=None):
    """Evaluate per-class binary metrics on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    samples = list(load_rs_samples(dataset_name, max_samples))
    print(f"Loaded {len(samples)} samples ({len(samples)//5} tiles × 5 classes)")

    # Group by class
    all_metrics = defaultdict(list)
    per_sample_results = []

    for sample in tqdm(samples, desc=dataset_name):
        img_size = sample.gt_mask.shape

        # Encode image (cached per tile — we re-encode each time for simplicity)
        inference_state = sam3.encode_image(sample.image)
        text_prompt = TEXT_PROMPTS[sample.class_name]

        # Get binary prediction
        pred_mask = sam3.predict_binary(inference_state, text_prompt)

        # Ensure 2D and resize if needed
        pred_mask = np.squeeze(pred_mask)
        if pred_mask.ndim != 2:
            pred_mask = pred_mask[..., 0] if pred_mask.ndim == 3 else pred_mask
        if pred_mask.shape[:2] != img_size[:2]:
            pred_mask = resize_mask(pred_mask, img_size[:2])

        # Compute metrics
        m = compute_all_metrics(pred_mask, sample.gt_mask)
        m['class'] = sample.class_name
        m['tile'] = sample.tile_name
        m['dataset'] = dataset_name
        all_metrics[sample.class_name].append(m)
        per_sample_results.append(m)

        # Save prediction visualization for first few tiles
        if output_dir and sample.tile_name in ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area15', 'top_potsdam_2_10']:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            from PIL import Image
            vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            vis[pred_mask > 0] = [0, 255, 0]     # green = prediction
            vis[sample.gt_mask > 0] = [255, 0, 0]  # red = GT (overlap = yellow)
            fname = f'{dataset_name}_{sample.tile_name}_{sample.class_name}.png'
            Image.fromarray(vis).save(os.path.join(vis_dir, fname))

    # Aggregate per class
    results = {}
    for cls_name in sorted(all_metrics.keys()):
        ms = all_metrics[cls_name]
        agg = {}
        for key in ['dice', 'iou', 'precision', 'recall']:
            vals = [m[key] for m in ms]
            agg[f'{key}_mean'] = np.mean(vals)
            agg[f'{key}_std'] = np.std(vals)
        results[cls_name] = agg
        print(f"  {cls_name:>10}: Dice={agg['dice_mean']:.3f}±{agg['dice_std']:.3f}  "
              f"IoU={agg['iou_mean']:.3f}±{agg['iou_std']:.3f}  "
              f"Prec={agg['precision_mean']:.3f}  Rec={agg['recall_mean']:.3f}")

    # Overall
    all_dice = [m['dice'] for m in per_sample_results]
    all_iou = [m['iou'] for m in per_sample_results]
    print(f"  {'OVERALL':>10}: mDice={np.mean(all_dice):.3f}  mIoU={np.mean(all_iou):.3f}")

    return results, per_sample_results, np.mean(all_dice), np.mean(all_iou)


def save_summary_csv(all_rows, output_path):
    """Save per-class summary to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','class','dice_mean','dice_std','iou_mean','iou_std',
                                                'precision_mean','recall_mean'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['vaihingen', 'potsdam'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--confidence', type=float, default=0.1)
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_dir, f'plan2_phase1_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print(f"Plan2 Phase 1: Per-Class Binary Baseline")
    print(f"Output: {output_root}")
    print(f"Confidence threshold: {args.confidence}")

    sam3 = SAM3Model(confidence_threshold=args.confidence)

    all_summary_rows = []

    for ds_name in args.datasets:
        results, per_sample, mDice, mIoU = evaluate_dataset(sam3, ds_name, args.max_samples, output_root)

        # Save per-dataset summary
        for cls_name, agg in results.items():
            all_summary_rows.append({
                'dataset': ds_name, 'class': cls_name,
                'dice_mean': round(agg['dice_mean'], 4),
                'dice_std': round(agg['dice_std'], 4),
                'iou_mean': round(agg['iou_mean'], 4),
                'iou_std': round(agg['iou_std'], 4),
                'precision_mean': round(agg['precision_mean'], 4),
                'recall_mean': round(agg['recall_mean'], 4),
            })

        # Save per-sample CSV
        csv_path = os.path.join(output_root, f'{ds_name}_per_sample.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset','tile','class','dice','iou','precision','recall'])
            writer.writeheader()
            for m in per_sample:
                writer.writerow({k: m[k] for k in ['dataset','tile','class','dice','iou','precision','recall']})

    # Save overall summary
    save_summary_csv(all_summary_rows, os.path.join(output_root, 'class_summary.csv'))

    # Save as JSON
    json.dump(all_summary_rows, open(os.path.join(output_root, 'results.json'), 'w'), indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Class':<12} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>10}")
    print("-" * 62)
    for row in all_summary_rows:
        print(f"{row['dataset']:<12} {row['class']:<12} {row['dice_mean']:>7.3f} {row['iou_mean']:>7.3f} "
              f"{row['precision_mean']:>9.3f} {row['recall_mean']:>9.3f}")

    sam3.cleanup()
    print(f"\nAll output: {output_root}")


if __name__ == '__main__':
    main()
