"""Shared semantic segmentation metrics (Kappa, etc.).

Import from any plan's eval script:
    import sys; sys.path.insert(0, "/root/Mynet/Personal-Project")
    from metrics_utils import compute_kappa, compute_kappa_from_cm
"""

import numpy as np


def compute_kappa(per_class_tp, per_class_gts, per_class_preds=None, per_class_union=None):
    """Cohen's Kappa from per-class TP / ground-truth / prediction counts.

    Call with either:
      - per_class_preds: kappa_from_tp_gt_pred(tp, gt, pred)
      - per_class_union:  kappa_from_tp_gt_union(tp, gt, union) — pred = union - gt + tp

    Args:
        per_class_tp:   [C] inter / true-positive per class
        per_class_gts:  [C] ground-truth pixel count per class (= TP + FN)
        per_class_preds: [C] prediction pixel count per class (= TP + FP). Optional if union given.
        per_class_union: [C] union per class (= TP + FP + FN). Optional if pred given.

    Returns:
        float: Cohen's Kappa ∈ [-1, 1].
    """
    tp = np.asarray(per_class_tp, dtype=np.float64)
    gt = np.asarray(per_class_gts, dtype=np.float64)

    if per_class_preds is None and per_class_union is not None:
        union = np.asarray(per_class_union, dtype=np.float64)
        pred = union - gt + tp
    elif per_class_preds is not None:
        pred = np.asarray(per_class_preds, dtype=np.float64)
    else:
        raise ValueError("Provide per_class_preds or per_class_union")

    total = gt.sum()
    if total == 0:
        return 0.0

    pa = tp.sum() / total
    pe = (gt * pred).sum() / (total * total)
    if pe >= 1.0:
        return 0.0
    return float((pa - pe) / (1.0 - pe))


def compute_kappa_from_cm(cm):
    """Cohen's Kappa from a confusion matrix.

    Args:
        cm: [C, C] confusion matrix (rows = GT, cols = prediction), integer counts.

    Returns:
        float: Cohen's Kappa ∈ [-1, 1].
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    pa = np.trace(cm) / total
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = (row_sum * col_sum).sum() / (total * total)
    if pe >= 1.0:
        return 0.0
    return float((pa - pe) / (1.0 - pe))
