"""
二分类分割指标: Dice, IoU, Precision, Recall.
基于 Medical-SAM3 的 metrics.py 改编.
"""

import numpy as np


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * intersection / max(denom, 1)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU / Jaccard index."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / max(union, 1)


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    return tp / max(tp + fp, 1)


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return tp / max(tp + fn, 1)


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all metrics for one sample."""
    return {
        'dice': compute_dice(pred, gt),
        'iou': compute_iou(pred, gt),
        'precision': compute_precision(pred, gt),
        'recall': compute_recall(pred, gt),
    }


def aggregate_class_metrics(per_sample_metrics: list) -> dict:
    """Aggregate per-sample metrics into per-class mean/std."""
    if not per_sample_metrics:
        return {}
    keys = per_sample_metrics[0].keys()
    result = {}
    for key in keys:
        vals = [m[key] for m in per_sample_metrics]
        result[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
    return result


def resize_mask(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize binary mask to target (H, W)."""
    from PIL import Image
    mask = np.squeeze(mask)
    h, w = int(target_shape[0]), int(target_shape[1])
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode='L')
    mask_resized = mask_img.resize((w, h), Image.NEAREST)
    return (np.array(mask_resized) > 127).astype(np.uint8)
