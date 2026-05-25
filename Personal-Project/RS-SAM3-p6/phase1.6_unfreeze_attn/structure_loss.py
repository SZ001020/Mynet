"""
Plan3 Route A: Structure Loss — 边缘加权 BCE + 加权 IoU

边界像素获得 ×(1+5×|avg_pool - original|) 的权重，
对 building/road 等具有清晰边界的类别特别有效。
"""

import torch
import torch.nn.functional as F


def structure_loss(pred_logits: torch.Tensor, gt_mask: torch.Tensor,
                   edge_weight: float = 5.0, pool_size: int = 31,
                   smooth: float = 1.0) -> torch.Tensor:
    """Edge-weighted BCE + weighted IoU loss for binary segmentation.

    Args:
        pred_logits: (B, C, H, W) per-class logits
        gt_mask: (B, H, W) long tensor with class indices (ignore_index=255)
        edge_weight: multiplier for boundary pixels (default 5)
        pool_size: avg_pool kernel size for edge detection (default 31)
        smooth: smoothing constant for IoU (default 1)

    Returns:
        scalar loss
    """
    B, C, H, W = pred_logits.shape

    # Build per-class binary targets: (B, C, H, W)
    gt_binary = torch.zeros(B, C, H, W, device=pred_logits.device, dtype=torch.float32)
    valid_mask = (gt_mask != 255)
    for c in range(C):
        gt_binary[:, c][valid_mask & (gt_mask == c)] = 1.0

    # Edge weight map: high at boundaries, low in interior
    # |avg_pool(mask) - mask| → 1 at edges, 0 in homogeneous regions
    edge_map = torch.abs(
        F.avg_pool2d(gt_binary, pool_size, stride=1, padding=pool_size // 2) - gt_binary
    )
    weit = 1.0 + edge_weight * edge_map  # (B, C, H, W)

    # Weighted BCE
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_binary, reduction='none')
    wbce = (weit * bce * valid_mask.unsqueeze(1).float()).sum() / (weit * valid_mask.unsqueeze(1).float()).sum().clamp(min=1e-8)

    # Weighted IoU
    pred = torch.sigmoid(pred_logits)
    inter = ((pred * gt_binary) * weit * valid_mask.unsqueeze(1).float()).sum(dim=(2, 3))
    union = (((pred + gt_binary) * weit * valid_mask.unsqueeze(1).float())).sum(dim=(2, 3))
    wiou = 1.0 - (inter + smooth) / (union - inter + smooth)
    wiou = wiou.mean()

    return wbce + wiou
