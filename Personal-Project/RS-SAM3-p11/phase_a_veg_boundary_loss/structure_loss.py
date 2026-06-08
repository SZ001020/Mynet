"""
P11-A: Structure Loss with tree/grass boundary extra weighting.

Modification from Plan7-A structure_loss:
- Added veg_boundary_weight parameter (default 3.0)
- Detects tree-grass boundary regions via avg_pool coexistence
- Applies extra weight to grass (c=2) and tree (c=3) channels at vegetation boundaries
"""

import torch
import torch.nn.functional as F


def structure_loss(pred_logits: torch.Tensor, gt_mask: torch.Tensor,
                   edge_weight: float = 5.0, pool_size: int = 31,
                   veg_boundary_weight: float = 3.0,
                   smooth: float = 1.0) -> torch.Tensor:
    """Edge-weighted BCE + weighted IoU loss with vegetation boundary boost.

    Args:
        pred_logits: (B, C, H, W) per-class logits
        gt_mask: (B, H, W) long tensor with class indices (ignore_index=255)
        edge_weight: multiplier for boundary pixels (default 5)
        pool_size: avg_pool kernel size for edge detection (default 31)
        veg_boundary_weight: extra weight multiplier at tree-grass boundaries (default 3.0)
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
    edge_map = torch.abs(
        F.avg_pool2d(gt_binary, pool_size, stride=1, padding=pool_size // 2) - gt_binary
    )
    weit = 1.0 + edge_weight * edge_map  # (B, C, H, W)

    # P11-A: Tree-grass boundary extra weighting
    # Detect where tree AND grass coexist within the pooling neighborhood
    if veg_boundary_weight > 0 and C >= 4:
        tree_mask = gt_binary[:, 3:4]   # (B, 1, H, W)
        grass_mask = gt_binary[:, 2:3]  # (B, 1, H, W)

        tree_nearby = (F.avg_pool2d(tree_mask, pool_size, stride=1,
                                     padding=pool_size // 2) > 0.01).float()
        grass_nearby = (F.avg_pool2d(grass_mask, pool_size, stride=1,
                                      padding=pool_size // 2) > 0.01).float()
        veg_boundary = tree_nearby * grass_nearby  # 1 where both coexist

        # Amplify edge weights for grass (c=2) and tree (c=3) at veg boundaries only
        veg_boost = 1.0 + veg_boundary_weight * veg_boundary  # (B, 1, H, W)
        weit[:, 2:4] = weit[:, 2:4] * veg_boost

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
