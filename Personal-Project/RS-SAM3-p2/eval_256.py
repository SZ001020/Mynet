#!/usr/bin/env python3
"""
Plan2 统一标准重评估：SAM3 per-class binary + 256² 滑动窗口 + 阈值调优

与 Plan3+ 一致的评估协议：
  - 256×256 滑动窗口，stride=128，overlap 平均
  - MFNet 标准测试集划分（Vaihingen 4 张，Potsdam 6 张）
  - 四个统一指标：OA, mIoU, per-class IoU, per-class OA
  - 额外：逐类二值指标（Dice, IoU, Precision, Recall）

对应原 Plan2 的 per-class binary pipeline：
  - Instance masks (top 10) OR semantic sigmoid top-K%
  - 阈值在 calibration tiles 上调优

用法:
  python eval_256.py --dataset vaihingen          # 只评估
  python eval_256.py --dataset vaihingen --tune    # 先调优阈值再评估
  python eval_256.py --dataset both --tune         # 两个数据集都做
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict

BASE = '/root/Mynet'
SE = f'{BASE}/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM3-p5')  # for dataset_adapter

from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL, VAIHINGEN_TRAIN, POTSDAM_TRAIN

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
IGNORE_INDEX = 255

# Calibration tiles (from training set, NOT test set)
# 与原始 Plan2 的 VAL_TILES 对齐（都在训练集中）
CALIB_TILES_VAHINGEN = [
    'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32',
    'top_mosaic_09cm_area34', 'top_mosaic_09cm_area37',
]
CALIB_TILES_POTSDAM = [
    'top_potsdam_6_7', 'top_potsdam_6_8', 'top_potsdam_6_9',
    'top_potsdam_7_7', 'top_potsdam_7_8', 'top_potsdam_7_9',
]

# K values for threshold sweep
K_VALUES = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]


def load_sam3_processor():
    _prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    processor = Sam3Processor(model, confidence_threshold=0.5, device='cuda')
    os.chdir(_prev)
    return processor


@torch.no_grad()
def predict_window_plan2(processor, patch_pil, k_pct=None):
    """Plan2 风格 per-class binary 推理（instance OR semantic top-K%）。

    对每个类独立推理，返回：(per_class_prob, per_class_binary)

    Args:
        processor: Sam3Processor
        patch_pil: 256×256 PIL RGB
        k_pct: top-K% 阈值，可以是 int（所有类统一）或 dict {cls_name: k}

    Returns:
        prob_maps: (5, H, W) float32 continuous confidence [0,1]
        bin_maps:  (5, H, W) uint8 binary masks
    """
    w, h = patch_pil.size
    if k_pct is None:
        k_pct = 15
    prob_maps = torch.zeros((5, h, w), device='cuda')
    bin_maps = torch.zeros((5, h, w), dtype=torch.uint8, device='cuda')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        inference_state = processor.set_image(patch_pil)

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            processor.reset_all_prompts(inference_state)
            state = processor.set_text_prompt(
                state=inference_state, prompt=cls_name)

            # per-class K value
            if isinstance(k_pct, dict):
                k_val = k_pct.get(cls_name, 15)
            else:
                k_val = k_pct

            # Instance masks (binary, OR'd)
            masks = state.get('masks')
            combined_bin = torch.zeros((h, w), dtype=torch.bool, device='cuda')
            if masks is not None and len(masks) > 0:
                for m in masks[:10]:
                    m_resized = m.squeeze()
                    if m_resized.shape != (h, w):
                        m_resized = F.interpolate(
                            m_resized.float().view(1, 1, *m_resized.shape),
                            size=(h, w), mode='nearest').squeeze() > 0.5
                    combined_bin = combined_bin | m_resized.bool()

            # Semantic head sigmoid → continuous probability
            sem = state.get('semantic_mask_logits')
            sem_prob = torch.zeros((h, w), device='cuda')
            if sem is not None:
                sem_prob = sem.squeeze().sigmoid().float()
                if sem_prob.dim() > 2:
                    sem_prob = sem_prob.squeeze()
                if sem_prob.shape != (h, w):
                    sem_prob = F.interpolate(
                        sem_prob.unsqueeze(0).unsqueeze(0),
                        size=(h, w), mode='bilinear',
                        align_corners=False).squeeze()

                # Top-K% threshold for binary
                k = max(int(sem_prob.numel() * k_val / 100.0), 50)
                thresh = sem_prob.flatten().topk(k).values[-1].item()
                sem_bin = sem_prob > thresh
                combined_bin = combined_bin | sem_bin

            # Continuous confidence: max of instance signal + semantic
            inst_signal = combined_bin.float()
            prob_maps[cls_idx] = torch.max(sem_prob, inst_signal * 0.8)
            bin_maps[cls_idx] = combined_bin.byte()

    return prob_maps.cpu(), bin_maps.cpu()


def sliding_accumulate(processor, img_np, k_pct=None):
    """256² 滑动窗口，累积 per-class probability maps 和 binary masks。

    Returns:
        prob_acc:  (5, H, W) float64 accumulated continuous confidences
        bin_acc:   (5, H, W) float64 accumulated binary mask counts
        count:     (H, W) float64 window overlap counts
    """
    H, W = img_np.shape[:2]
    stride = 128
    prob_acc = np.zeros((5, H, W), dtype=np.float64)
    bin_acc = np.zeros((5, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    n_windows = 0

    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue

            patch_pil = Image.fromarray(img_np[y:y2, x:x2])
            prob, bin_mask = predict_window_plan2(processor, patch_pil, k_pct)

            m = 16
            im, jm = min(m, ph // 4), min(m, pw // 4)
            for c in range(5):
                prob_acc[c, y+im:y2-im, x+jm:x2-jm] += \
                    prob[c, im:ph-im, jm:pw-jm].numpy()
                bin_acc[c, y+im:y2-im, x+jm:x2-jm] += \
                    bin_mask[c, im:ph-im, jm:pw-jm].numpy()
            count[y+im:y2-im, x+jm:x2-jm] += 1.0
            n_windows += 1

    count[count == 0] = 1.0
    prob_acc /= count
    bin_acc /= count  # becomes fraction [0,1]
    return prob_acc, bin_acc, count, n_windows


def compute_multi_metrics(prob_acc, gt_np):
    """从 per-class probability maps 计算多类别指标（argmax）。"""
    pred = prob_acc.argmax(0).astype(np.int64)
    mask = gt_np != IGNORE_INDEX

    total_correct = (pred[mask] == gt_np[mask]).sum()
    total_pixels = mask.sum()

    per_class_iou = {}
    per_class_oa = {}
    for i, c in enumerate(CLASS_NAMES):
        pc = (pred == i)
        lc = (gt_np == i)
        inter = (pc & lc).sum()
        union = (pc | lc).sum()
        per_class_iou[c] = (inter / union * 100) if union > 0 else 0.0
        tn = ((~pc) & (~lc) & mask).sum()
        per_class_oa[c] = (inter + tn) / mask.sum() * 100

    return {
        'oa': total_correct / total_pixels * 100,
        'miou': np.mean(list(per_class_iou.values())),
        'per_class_iou': per_class_iou,
        'per_class_oa': per_class_oa,
    }


def compute_binary_metrics(bin_acc, gt_np, bin_threshold=0.3):
    """从二值累积图计算逐类二值指标。

    bin_acc[c] 是每个像素被预测为正类的窗口比例 [0,1]。
    bin_threshold: 超过此比例即判定为正类。
    """
    per_class_binary = {}
    for i, c in enumerate(CLASS_NAMES):
        pred = (bin_acc[i] > bin_threshold).astype(np.uint8)
        gt = (gt_np == i).astype(np.uint8)
        mask = gt_np != IGNORE_INDEX
        valid_pred = pred[mask]
        valid_gt = gt[mask]

        tp = (valid_pred & valid_gt).sum()
        fp = (valid_pred & ~valid_gt).sum()
        fn = (~valid_pred & valid_gt).sum()

        precision = tp / max(tp + fp, 1) * 100
        recall = tp / max(tp + fn, 1) * 100
        dice = 2 * tp / max(2 * tp + fp + fn, 1) * 100
        iou = tp / max(tp + fp + fn, 1) * 100

        per_class_binary[c] = {
            'dice': dice, 'iou': iou,
            'precision': precision, 'recall': recall,
        }

    return per_class_binary


def evaluate_tile(processor, tile, img_dir, gt_dir, img_suf, gt_suf, k_pct):
    """评估单张 tile（多类别 + 二值指标）。"""
    ip = f'{img_dir}/{tile}{img_suf}'
    gp = f'{gt_dir}/{tile}{gt_suf}'
    img = np.array(Image.open(ip).convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))

    prob_acc, bin_acc, _, n_windows = sliding_accumulate(processor, img, k_pct)
    multi = compute_multi_metrics(prob_acc, gt)
    binary = compute_binary_metrics(bin_acc, gt)
    return multi, binary, n_windows, k_pct


def calibrate_thresholds(processor, dataset_name):
    """在 calibration tiles 上扫描 K 值，找到最优 per-class top-K%。

    返回 best_k dict: {class_name: best_k_pct}
    """
    if dataset_name == 'vaihingen':
        tiles = CALIB_TILES_VAHINGEN
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        img_suf, gt_suf = '.tif', '.tif'
    else:
        tiles = CALIB_TILES_POTSDAM
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'

    print(f"  Calibrating on {len(tiles)} tiles: {tiles}")

    # For each K, accumulate per-class binary IoU across calibration tiles
    best_k = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"    {cls_name}: sweeping K...", end=' ', flush=True)
        best_iou = 0.0
        best_k_val = 15  # default

        for k in K_VALUES:
            total_inter = 0.0
            total_union = 0.0
            for tile in tiles:
                ip = f'{img_dir}/{tile}{img_suf}'
                gp = f'{gt_dir}/{tile}{gt_suf}'
                if not os.path.exists(ip) or not os.path.exists(gp):
                    continue
                img = np.array(Image.open(ip).convert('RGB'))
                gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))

                # 对每个 calibration tile 只用少量采样窗口加速
                # （完整 256² 滑动太慢，用 512² stride=384 粗采样）
                H, W = img.shape[:2]
                stride = 384
                inter_c, union_c = 0.0, 0.0
                n_samples = 0
                for y in range(0, H - 128, stride):
                    for x in range(0, W - 128, stride):
                        y2, x2 = min(y + 512, H), min(x + 512, W)
                        if y2 - y < 256 or x2 - x < 256:
                            continue
                        patch_pil = Image.fromarray(img[y:y2, x:x2])
                        _, bin_mask = predict_window_plan2(
                            processor, patch_pil, k)
                        pred_c = bin_mask[cls_idx].numpy()
                        gt_c = (gt[y:y2, x:x2] == cls_idx)
                        inter_c += (pred_c & gt_c).sum()
                        union_c += (pred_c | gt_c).sum()
                        n_samples += 1

                if n_samples > 0:
                    total_inter += inter_c
                    total_union += union_c

            if total_union > 0:
                iou = total_inter / total_union * 100
            else:
                iou = 0.0

            if iou > best_iou:
                best_iou = iou
                best_k_val = k

        best_k[cls_name] = best_k_val
        print(f"K={best_k_val}% (IoU={best_iou:.1f}%)")

    return best_k


def main():
    parser = argparse.ArgumentParser(
        description='Plan2 统一标准重评估: SAM3 per-class binary + 256² SW')
    parser.add_argument('--dataset', default='vaihingen',
                        choices=['vaihingen', 'potsdam', 'both'])
    parser.add_argument('--tune', action='store_true',
                        help='在 calibration tiles 上调优 per-class 阈值')
    parser.add_argument('--output', default='/root/autodl-tmp/runs')
    args = parser.parse_args()

    print("=" * 60)
    print("Plan2 统一标准重评估")
    print("  协议: SAM3 per-class binary (instance OR semantic top-K%)")
    print("  窗口: 256² sliding, stride=128, overlap avg")
    print("  测试集: MFNet standard split")
    print("  指标: 多类别 (OA/mIoU) + 二值 (Dice/IoU/P/R)")
    print("=" * 60)

    print("\n[1/3] Loading SAM3 model...")
    t0 = time.time()
    processor = load_sam3_processor()
    print(f"  Loaded in {time.time() - t0:.0f}s")

    datasets = ['vaihingen', 'potsdam'] if args.dataset == 'both' else [args.dataset]

    # Per-dataset threshold tuning
    all_best_k = {}
    if args.tune:
        print(f"\n[2/3] Threshold calibration...")
        for ds in datasets:
            print(f"\n  {ds.upper()}:")
            all_best_k[ds] = calibrate_thresholds(processor, ds)
        # Save tuning results
        ts = time.strftime('%Y%m%d_%H%M%S')
        json.dump(all_best_k, open(
            os.path.join(args.output, f'plan2_thresholds_{ts}.json'), 'w'), indent=2)
    else:
        # Use Plan2 original defaults
        for ds in datasets:
            all_best_k[ds] = {c: 15 for c in CLASS_NAMES}

    # Evaluation
    print(f"\n[{'3' if args.tune else '2'}/{'3' if args.tune else '2'}] "
          f"256² sliding window evaluation...")

    all_results = {}
    for ds in datasets:
        tiles = VAIHINGEN_VAL if ds == 'vaihingen' else POTSDAM_VAL
        img_dir = ('/root/autodl-tmp/dataset/Vaihingen/top' if ds == 'vaihingen'
                   else '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB')
        gt_dir = ('/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
                  if ds == 'vaihingen'
                  else '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants')
        img_suf = '.tif' if ds == 'vaihingen' else '_RGB.tif'
        gt_suf = '.tif' if ds == 'vaihingen' else '_label.tif'

        best_k = all_best_k[ds]
        print(f"\n  {ds.upper()} (K={best_k}):")

        ds_multi = []
        ds_binary = defaultdict(list)
        total_windows = 0

        for tile in tiles:
            ip = f'{img_dir}/{tile}{img_suf}'
            if not os.path.exists(ip):
                print(f"    SKIP {tile}: file not found")
                continue
            print(f"    [{tile}]...", end=' ', flush=True)
            t0 = time.time()

            multi, binary, nw, _ = evaluate_tile(
                processor, tile, img_dir, gt_dir, img_suf, gt_suf, best_k)

            total_windows += nw
            ds_multi.append(multi)
            for c in CLASS_NAMES:
                for k, v in binary[c].items():
                    ds_binary[f'{c}_{k}'].append(v)

            print(f"{nw}w, {time.time()-t0:.0f}s, "
                  f"OA={multi['oa']:.1f}%, mIoU={multi['miou']:.1f}%")

        # Aggregate
        avg_oa = np.mean([m['oa'] for m in ds_multi])
        avg_miou = np.mean([m['miou'] for m in ds_multi])
        avg_pc_iou = {c: np.mean([m['per_class_iou'][c] for m in ds_multi])
                      for c in CLASS_NAMES}
        avg_pc_oa = {c: np.mean([m['per_class_oa'][c] for m in ds_multi])
                     for c in CLASS_NAMES}
        avg_binary = {}
        for c in CLASS_NAMES:
            avg_binary[c] = {
                'dice': np.mean(ds_binary[f'{c}_dice']),
                'iou': np.mean(ds_binary[f'{c}_iou']),
                'precision': np.mean(ds_binary[f'{c}_precision']),
                'recall': np.mean(ds_binary[f'{c}_recall']),
            }

        result = {
            'dataset': ds,
            'method': 'Plan2 (instance OR semantic top-K%)',
            'k_used': best_k,
            'per_class_best_k': best_k,
            'num_tiles': len(ds_multi),
            'total_windows': total_windows,
            'multi_class': {
                'avg_oa': avg_oa,
                'avg_miou': avg_miou,
                'per_class_iou': avg_pc_iou,
                'per_class_oa': avg_pc_oa,
            },
            'binary': avg_binary,
            'tile_details': ds_multi,
        }
        all_results[ds] = result

        print(f"\n  {'─'*50}")
        print(f"  {ds.upper()} 多类别指标 (argmax on per-class prob):")
        print(f"    OA   = {avg_oa:.2f}%")
        print(f"    mIoU = {avg_miou:.2f}%")
        pcs = ', '.join(f"{c}={avg_pc_iou[c]:.1f}" for c in CLASS_NAMES)
        print(f"    Per-class IoU: {pcs}")
        pcs = ', '.join(f"{c}={avg_pc_oa[c]:.1f}" for c in CLASS_NAMES)
        print(f"    Per-class OA : {pcs}")
        print(f"\n  {ds.upper()} 二值指标 (per-class, independent):")
        print(f"    {'Class':<10} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Rec':>7}")
        for c in CLASS_NAMES:
            b = avg_binary[c]
            print(f"    {c:<10} {b['dice']:>6.1f}% {b['iou']:>6.1f}% "
                  f"{b['precision']:>6.1f}% {b['recall']:>6.1f}%")
        print(f"{'─'*50}")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.output, f'plan2_eval256_{ts}.json')
    json.dump(all_results, open(out_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Cross-dataset summary
    if len(datasets) == 2:
        print(f"\n{'='*60}")
        print("Combined Summary:")
        combined_miou = (all_results['vaihingen']['multi_class']['avg_miou'] +
                         all_results['potsdam']['multi_class']['avg_miou']) / 2
        print(f"  Avg mIoU: {combined_miou:.2f}%")
        for c in CLASS_NAMES:
            avg_iou = (all_results['vaihingen']['multi_class']['per_class_iou'][c] +
                       all_results['potsdam']['multi_class']['per_class_iou'][c]) / 2
            avg_bin_iou = (all_results['vaihingen']['binary'][c]['iou'] +
                           all_results['potsdam']['binary'][c]['iou']) / 2
            print(f"    {c}: multi-IoU={avg_iou:.1f}%  binary-IoU={avg_bin_iou:.1f}%")

    del processor
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == '__main__':
    main()
