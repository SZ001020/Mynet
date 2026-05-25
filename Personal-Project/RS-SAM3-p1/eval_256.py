#!/usr/bin/env python3
"""
Plan1 统一标准重评估：SAM3 zero-shot + 256² 滑动窗口 + 多类别 argmax

与 Plan3+ 完全一致的评估协议：
  - 256×256 滑动窗口，stride=128，overlap 平均
  - MFNet 标准测试集划分（Vaihingen 4 张，Potsdam 6 张）
  - 四个指标：OA, mIoU, per-class IoU, per-class OA
  - 5 类 (no clutter)，单词语义文本提示

对应原 Plan1 Phase 1 的 Dual-Head (Instance+Semantic) + Presence Score 推理管线。
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = '/root/Mynet'
SE = f'{BASE}/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM3-p5')  # for dataset_adapter

from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
IGNORE_INDEX = 255


def load_sam3_processor(device='cuda'):
    """加载 SAM3 模型及其 Sam3Processor。"""
    _prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt',
        device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)
    os.chdir(_prev)
    return processor


@torch.no_grad()
def predict_window(processor, patch_pil):
    """在单个 256² PIL 图像上做 SAM3 zero-shot 多类别推理。

    Args:
        processor: Sam3Processor 实例
        patch_pil: 256×256 PIL RGB 图像

    Returns:
        seg_logits: (5, 256, 256) float32 tensor on CPU
    """
    w, h = patch_pil.size
    seg_logits = torch.zeros((5, h, w), device='cuda')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        inference_state = processor.set_image(patch_pil)

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=cls_name)

            # Instance head (Transformer decoder)
            masks_logits = inference_state.get('masks_logits')
            if masks_logits is not None and masks_logits.shape[0] > 0:
                scores = inference_state.get('object_score',
                              inference_state.get('scores'))
                for inst_id in range(masks_logits.shape[0]):
                    inst_logits = masks_logits[inst_id]  # (H', W') or (1, H', W')
                    if inst_logits.dim() == 3:
                        inst_logits = inst_logits.squeeze(0)
                    inst_score = (scores[inst_id].item() if scores is not None
                                  and scores.numel() > inst_id else 1.0)
                    if inst_logits.shape != (h, w):
                        inst_logits = F.interpolate(
                            inst_logits.float().view(1, 1, *inst_logits.shape),
                            size=(h, w), mode='bilinear',
                            align_corners=False).squeeze()
                    seg_logits[cls_idx] = torch.max(
                        seg_logits[cls_idx], inst_logits * inst_score)

            # Semantic head
            semantic_logits = inference_state.get('semantic_mask_logits')
            if semantic_logits is not None:
                # Handle variable dims (matching original segmentor logic)
                if semantic_logits.dim() == 2:
                    semantic_logits = semantic_logits.unsqueeze(0).unsqueeze(0)
                elif semantic_logits.dim() == 3:
                    semantic_logits = semantic_logits.unsqueeze(0)
                # Now semantic_logits is (1, 1, H', W') or (1, C, H', W')
                if semantic_logits.shape[-2:] != (h, w):
                    semantic_logits = F.interpolate(
                        semantic_logits.float(),
                        size=(h, w), mode='bilinear',
                        align_corners=False)
                semantic_logits = semantic_logits.squeeze()
                seg_logits[cls_idx] = torch.max(
                    seg_logits[cls_idx], semantic_logits.float())

            # Presence score
            ps = inference_state.get('presence_score')
            if ps is not None and isinstance(ps, torch.Tensor):
                seg_logits[cls_idx] = seg_logits[cls_idx] * ps.squeeze()

    return seg_logits.cpu()


def sliding_eval(processor, img_np, gt_np, verbose=True):
    """256² 滑动窗口评估单张 tile。

    Args:
        processor: Sam3Processor
        img_np: (H, W, 3) uint8 RGB 图像
        gt_np: (H, W) int64 标签

    Returns:
        dict with oa, miou, per_class_iou, per_class_oa
    """
    H, W = img_np.shape[:2]
    stride = 128
    pred_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    n_windows = 0
    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue

            crop = img_np[y:y2, x:x2]
            patch_pil = Image.fromarray(crop)
            seg_logits = predict_window(processor, patch_pil)  # (5, ph, pw)
            pred = seg_logits.argmax(0).numpy().astype(np.float64)

            m = 16
            im, jm = min(m, ph // 4), min(m, pw // 4)
            pred_sum[y + im:y2 - im, x + jm:x2 - jm] += pred[im:ph - im, jm:pw - jm]
            count[y + im:y2 - im, x + jm:x2 - jm] += 1.0
            n_windows += 1

    count[count == 0] = 1.0
    pred = np.round(pred_sum / count).astype(np.int64)

    # 计算指标
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
        # per-class OA: (TP + TN) / total valid pixels
        tn = ((~pc) & (~lc) & mask).sum()
        per_class_oa[c] = (inter + tn) / mask.sum() * 100

    result = {
        'oa': total_correct / total_pixels * 100,
        'miou': np.mean(list(per_class_iou.values())),
        'per_class_iou': per_class_iou,
        'per_class_oa': per_class_oa,
        'n_windows': n_windows,
    }
    if verbose:
        print(f"    {n_windows} windows, OA={result['oa']:.1f}%, "
              f"mIoU={result['miou']:.1f}%")
        for c in CLASS_NAMES:
            print(f"      {c}: IoU={per_class_iou[c]:.1f}%  "
                  f"OA={per_class_oa[c]:.1f}%")

    return result


def evaluate_dataset(processor, dataset_name):
    """评估整个数据集 (vaihingen 或 potsdam)。"""
    if dataset_name == 'vaihingen':
        tiles = VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        img_suf = '.tif'
        gt_suf = '.tif'
    else:
        tiles = POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        img_suf = '_RGB.tif'
        gt_suf = '_label.tif'

    tile_results = []
    for tile in tiles:
        ip = f'{img_dir}/{tile}{img_suf}'
        gp = f'{gt_dir}/{tile}{gt_suf}'
        if not os.path.exists(ip) or not os.path.exists(gp):
            print(f"  SKIP {tile}: file not found")
            continue

        img = np.array(Image.open(ip).convert('RGB'))
        gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))
        print(f"  [{tile}] {img.shape[1]}×{img.shape[0]}...")
        result = sliding_eval(processor, img, gt)
        result['tile'] = tile
        tile_results.append(result)

    # 汇总平均（按 tile 平均，每个 tile 等权重）
    avg_oa = np.mean([r['oa'] for r in tile_results])
    avg_miou = np.mean([r['miou'] for r in tile_results])
    avg_pc_iou = {c: np.mean([r['per_class_iou'][c] for r in tile_results])
                  for c in CLASS_NAMES}
    avg_pc_oa = {c: np.mean([r['per_class_oa'][c] for r in tile_results])
                 for c in CLASS_NAMES}
    total_windows = sum(r['n_windows'] for r in tile_results)

    return {
        'dataset': dataset_name,
        'num_tiles': len(tile_results),
        'total_windows': total_windows,
        'avg_oa': avg_oa,
        'avg_miou': avg_miou,
        'per_class_iou': avg_pc_iou,
        'per_class_oa': avg_pc_oa,
        'tile_results': tile_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Plan1 统一标准重评估: SAM3 zero-shot + 256² 滑动窗口')
    parser.add_argument('--dataset', default='vaihingen',
                        choices=['vaihingen', 'potsdam', 'both'])
    parser.add_argument('--output', default='/root/autodl-tmp/runs',
                        help='输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print("Plan1 统一标准重评估")
    print("  协议: SAM3 zero-shot (Instance+Semantic+Presence)")
    print("  窗口: 256² sliding, stride=128, overlap avg")
    print("  测试集: MFNet standard split")
    print("=" * 60)

    print("\n[1/2] Loading SAM3 model + Sam3Processor...")
    t0 = time.time()
    processor = load_sam3_processor()
    print(f"  Loaded in {time.time() - t0:.0f}s")

    datasets = ['vaihingen', 'potsdam'] if args.dataset == 'both' else [args.dataset]
    all_results = {}

    for ds in datasets:
        print(f"\n[2/2] Evaluating {ds.upper()}...")
        t0 = time.time()
        result = evaluate_dataset(processor, ds)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s ({result['total_windows']} windows, "
              f"{elapsed / max(result['total_windows'], 1):.2f}s/window)")

        print(f"\n{'─' * 50}")
        print(f"  {ds.upper()} Summary:")
        print(f"    OA   = {result['avg_oa']:.2f}%")
        print(f"    mIoU = {result['avg_miou']:.2f}%")
        pcs = ', '.join(f"{c}={result['per_class_iou'][c]:.1f}"
                        for c in CLASS_NAMES)
        print(f"    Per-class IoU: {pcs}")
        pcs = ', '.join(f"{c}={result['per_class_oa'][c]:.1f}"
                        for c in CLASS_NAMES)
        print(f"    Per-class OA : {pcs}")
        print(f"{'─' * 50}")

        all_results[ds] = {
            'avg_oa': result['avg_oa'],
            'avg_miou': result['avg_miou'],
            'per_class_iou': result['per_class_iou'],
            'per_class_oa': result['per_class_oa'],
        }

        # 保存 tile 级结果
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(args.output,
                                f'plan1_eval256_{ds}_{ts}.json')
        json.dump(result, open(out_path, 'w'), indent=2, default=str)
        print(f"  Results saved to {out_path}")

    # 汇总
    if len(datasets) == 2:
        print(f"\n{'=' * 60}")
        print("Combined Summary (Vaihingen + Potsdam):")
        combined_miou = (all_results['vaihingen']['avg_miou'] +
                         all_results['potsdam']['avg_miou']) / 2
        print(f"  Avg mIoU across datasets: {combined_miou:.2f}%")
        for c in CLASS_NAMES:
            avg_c = (all_results['vaihingen']['per_class_iou'][c] +
                     all_results['potsdam']['per_class_iou'][c]) / 2
            print(f"    {c}: avg IoU = {avg_c:.1f}%")

    # 清理
    del processor
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == '__main__':
    main()
