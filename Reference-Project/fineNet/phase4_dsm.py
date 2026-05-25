"""
Phase 4: 多模态 SAM 3 — DSM 高程数据注入

方式 A: DSM → Geometric Prompt 注入（零样本，无训练）
方式 B: DSM 辅助训练信号（基于 Phase 3 微调框架扩展）

DSM 处理: 计算 nDSM → 提取高于地面的连通域 → 生成 box prompts
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os, sys, json, time, gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, '/root/Mynet/sam3-main')
sys.path.insert(0, '/root/Mynet/fineNet')

from segearthov3_segmentor import SegEarthOV3Segmentation, get_cls_idx
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================
# DSM → Geometric Prompt 转换
# ============================================================

def compute_ndsm(dsm, ground_kernel_size=101):
    """
    计算归一化数字表面模型 (nDSM = DSM - 地面高程).
    地面高程通过大核形态学开运算估计。
    """
    from scipy.ndimage import grey_opening
    ground = grey_opening(dsm, size=ground_kernel_size)
    return dsm - ground


def extract_object_boxes(ndsm, min_height=2.0, max_height=50.0,
                          min_area=100, max_area=50000):
    """
    从 nDSM 中提取高于地面的物体 box。
    Args:
        ndsm: (H, W) 归一化高程 (米)
        min_height: 最小物体高度 (m)
        max_height: 最大物体高度
        min_area: 最小物体面积 (px)
        max_area: 最大物体面积
    Returns:
        boxes: [[cx, cy, w, h], ...] 归一化到 [0,1]
    """
    mask = (ndsm > min_height) & (ndsm < max_height)
    labeled, num = ndimage.label(mask)

    boxes = []
    for label_id in range(1, num + 1):
        region = (labeled == label_id)
        area = region.sum()
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(region)
        if len(ys) < 4:
            continue

        h, w = ndsm.shape
        cy = ys.mean() / h
        cx = xs.mean() / w
        bw = (xs.max() - xs.min()) / w
        bh = (ys.max() - ys.min()) / h

        boxes.append([cx, cy, bw, bh])

    return boxes


# ============================================================
# 方式 A: Zero-shot DSM 注入推理
# ============================================================

class DSMEnhancedSegmentor:
    """
    包装 SegEarthOV3Segmentation，在推理时注入 DSM geometric prompts。
    """
    def __init__(self, classname_path, prob_thd=0.1, bg_idx=0,
                 confidence_threshold=0.4, device='cuda'):
        self.device = torch.device(device)

        model = build_sam3_image_model(
            bpe_path=f'{PROJECT_ROOT}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path=f'{PROJECT_ROOT}/weights/sam3/sam3.pt',
            device=device,
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx

        # DSM box cache (per-image)
        self.dsm_boxes = None

    def set_dsm_boxes(self, boxes):
        """预计算 DSM boxes 并缓存."""
        self.dsm_boxes = boxes

    @torch.no_grad()
    def _inference_single_view(self, image, dsm_prior=None):
        """
        Args:
            image: PIL Image
            dsm_prior: (H, W) float [0,1] mask where higher values = more likely object
        """
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state, prompt=query_word)

                # Instance masks
                if inference_state.get('masks_logits') is not None and \
                   inference_state['masks_logits'].shape[0] > 0:
                    inst_len = min(inference_state['masks_logits'].shape[0], 10)
                    for inst_id in range(inst_len):
                        instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                        score = inference_state.get('object_score',
                                inference_state.get('scores', torch.ones(1, device=self.device)))
                        if isinstance(score, torch.Tensor) and score.numel() > inst_id:
                            s = score[inst_id]
                        elif isinstance(score, torch.Tensor):
                            s = score.squeeze()
                        else:
                            s = 1.0
                        if instance_logits.shape != (h, w):
                            instance_logits = F.interpolate(
                                instance_logits.view(1, 1, *instance_logits.shape),
                                size=(h, w), mode='bilinear', align_corners=False).squeeze()
                        seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * s)

                # Semantic mask
                semantic_logits = inference_state.get('semantic_mask_logits')
                if semantic_logits is not None:
                    if semantic_logits.dim() == 2:
                        semantic_logits = semantic_logits.unsqueeze(0)
                    if semantic_logits.dim() == 3:
                        semantic_logits = semantic_logits.unsqueeze(0)
                    if semantic_logits.shape[-2:] != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits, size=(h, w), mode='bilinear', align_corners=False)
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits.squeeze())

                # === DSM prior: boost logits in above-ground regions for object classes ===
                if dsm_prior is not None:
                    dsm_tensor = torch.from_numpy(dsm_prior).float().to(self.device)
                    if dsm_tensor.shape != (h, w):
                        dsm_tensor = F.interpolate(
                            dsm_tensor.view(1, 1, *dsm_tensor.shape),
                            size=(h, w), mode='bilinear', align_corners=False).squeeze()
                    # Apply DSM boost for building, tree, car (indices 1,3,4)
                    object_classes = [1, 3, 4]  # building, tree, car
                    if query_idx in object_classes:
                        boost = 1.0 + dsm_tensor * 3.0  # up to 4x in high areas
                        seg_logits[query_idx] = seg_logits[query_idx] * boost
                    # Apply DSM dampening for road, grass (indices 0,2) — lower in elevated areas
                    ground_classes = [0, 2]  # road, grass
                    if query_idx in ground_classes:
                        dampen = 1.0 / (1.0 + dsm_tensor * 5.0)  # reduce in high areas
                        seg_logits[query_idx] = seg_logits[query_idx] * dampen

                # Presence score
                presence = inference_state.get('presence_score')
                if presence is not None and isinstance(presence, torch.Tensor):
                    seg_logits[query_idx] = seg_logits[query_idx] * presence.squeeze()

        return seg_logits

    def predict_image(self, image_path, dsm_path=None):
        image = Image.open(image_path).convert('RGB')
        max_edge = 2000
        if max(image.size) > max_edge:
            ratio = max_edge / max(image.size)
            image = image.resize(
                (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                Image.BILINEAR)

        # Load DSM prior
        dsm_prior = None
        if dsm_path and os.path.exists(dsm_path):
            dsm = np.array(Image.open(dsm_path))
            # resize DSM to match image
            dsm_img = Image.fromarray(dsm)
            if dsm_img.size != image.size:
                dsm_img = dsm_img.resize(image.size, Image.BILINEAR)
            dsm_resized = np.array(dsm_img)
            ndsm = compute_ndsm(dsm_resized)
            # Normalize to [0,1]: height 0m=0, height 10m=1
            dsm_prior = np.clip(ndsm / 10.0, 0, 1).astype(np.float32)

        seg_logits = self._inference_single_view(image, dsm_prior)

        if self.num_cls != self.num_queries:
            seg_logits = seg_logits.unsqueeze(0)
            cls_index = F.one_hot(self.query_idx)
            cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
            seg_logits = (seg_logits * cls_index.to(self.device)).max(1)[0]

        seg_pred = seg_logits.argmax(0).cpu().numpy()
        max_vals = seg_logits.max(0)[0].cpu().numpy()
        seg_pred[max_vals < self.prob_thd] = self.bg_idx

        return seg_pred

    def cleanup(self):
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 方式 B: DSM 增强微调
# ============================================================

class DSMAugmentedDataset:
    """
    包装训练数据集，额外提供 DSM box prompts。
    在 Phase 3 数据集基础上增加 DSM 特征通道或 box prompts。
    """

    @staticmethod
    def get_dsm_path(rgb_path, dataset_name):
        """从 RGB 路径推断 DSM 路径."""
        if dataset_name == 'vaihingen':
            tile = os.path.basename(rgb_path).replace('.tif', '')
            tile = tile.replace('top_mosaic_09cm_', 'dsm_09cm_matching_')
            return os.path.join('/root/autodl-tmp/dataset/Vaihingen/dsm', f'{tile}.tif')
        elif dataset_name == 'potsdam':
            tile = os.path.basename(rgb_path).replace('_RGB.tif', '')
            tile = tile.replace('top_potsdam_', 'dsm_potsdam_')
            return os.path.join('/root/autodl-tmp/dataset/Potsdam/1_DSM', f'{tile}.tif')
        return None

    @staticmethod
    def precompute_boxes_for_dataset(tile_list, dataset_name, data_root):
        """为数据集预计算所有 DSM boxes（一次计算，多次使用）."""
        all_boxes = {}
        for tile in tile_list:
            if dataset_name == 'vaihingen':
                img_path = os.path.join(data_root, 'top', f'{tile}.tif')
            else:
                img_path = os.path.join(data_root, '2_Ortho_RGB', f'{tile}_RGB.tif')

            dsm_path = DSMAugmentedDataset.get_dsm_path(img_path, dataset_name)
            if dsm_path and os.path.exists(dsm_path):
                dsm = np.array(Image.open(dsm_path))
                ndsm = compute_ndsm(dsm)
                boxes = extract_object_boxes(ndsm, min_height=2.0)
                all_boxes[tile] = boxes
            else:
                all_boxes[tile] = []

        return all_boxes


# ============================================================
# 评估函数
# ============================================================

def evaluate_on_dataset(segmentor, config, dataset_name, output_dir, use_dsm):
    """运行完整评估."""
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.logging import MMLogger
    import custom_datasets  # noqa

    logger = MMLogger.get_instance('phase4', log_level='INFO')

    cfg = Config.fromfile(config)
    cfg.model.use_transformer_decoder = True
    cfg.model.use_sem_seg = True
    cfg.model.use_presence_score = False

    # Build runner for data loading
    cfg.work_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect validation images
    val_file = cfg.test_dataloader.dataset.ann_file
    img_dir = os.path.join(cfg.test_dataloader.dataset.data_root,
                           cfg.test_dataloader.dataset.data_prefix['img_path'])
    img_suffix = cfg.test_dataloader.dataset.img_suffix

    with open(val_file, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]

    # Compute per-image metrics
    total_intersect = np.zeros(5)
    total_union = np.zeros(5)
    total_correct = 0
    total_pixels = 0

    for i, tile in enumerate(tiles):
        img_path = os.path.join(img_dir, f'{tile}{img_suffix}')

        # DSM path
        if use_dsm:
            dsm_path = DSMAugmentedDataset.get_dsm_path(img_path, dataset_name)
        else:
            dsm_path = None

        # Inference
        pred = segmentor.predict_image(img_path, dsm_path)

        # Compare with GT
        if dataset_name == 'vaihingen':
            gt_path = img_path.replace('/top/', '/gts_index/').replace('.tif', '.png')
        else:
            gt_path = img_path.replace('2_Ortho_RGB', 'labels_index').replace('_RGB.tif', '.png')
        if os.path.exists(gt_path):
            gt = np.array(Image.open(gt_path))
            # Resize GT to match prediction size
            if gt.shape[:2] != pred.shape[:2]:
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                    (pred.shape[1], pred.shape[0]), Image.NEAREST))
            # Remap 1-6 → 0-4 (clutter→255)
            gt_mapped = np.full_like(gt, 255, dtype=np.int64)
            for orig, new in {1:0,2:1,3:2,4:3,5:4,6:255}.items():
                gt_mapped[gt == orig] = new

            for c in range(5):
                pred_c = (pred == c)
                gt_c = (gt_mapped == c)
                total_intersect[c] += (pred_c & gt_c).sum()
                total_union[c] += (pred_c | gt_c).sum()

            mask = (gt_mapped != 255)
            total_correct += (pred[mask] == gt_mapped[mask]).sum()
            total_pixels += mask.sum()

        print(f'  [{i+1}/{len(tiles)}] {tile}', end='\r')

    # Compute metrics
    per_class_iou = []
    for c in range(5):
        iou = total_intersect[c] / max(total_union[c], 1) * 100
        per_class_iou.append(iou)
    miou = np.mean(per_class_iou)
    aAcc = total_correct / max(total_pixels, 1) * 100

    return aAcc, miou, per_class_iou


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--dataset', choices=['vaihingen', 'potsdam'], default='vaihingen')
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_dir, f'phase4_dsm_{args.mode}_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # Config paths (no-clutter versions)
    configs = {
        'vaihingen': f'{PROJECT_ROOT}/configs/cfg_vaihingen_noclutter.py',
        'potsdam': f'{PROJECT_ROOT}/configs/cfg_potsdam_noclutter.py',
    }

    results = {}

    # === Baseline (no DSM) ===
    print(f"\n{'='*60}")
    print(f"Baseline (no DSM) — {args.dataset}")
    seg = DSMEnhancedSegmentor(
        classname_path=f'{PROJECT_ROOT}/configs/cls_{args.dataset}_noclutter.txt',
        prob_thd=0.1, bg_idx=0, confidence_threshold=0.4)
    seg.set_dsm_boxes([])
    out_dir = os.path.join(output_root, f'{args.dataset}_baseline')
    aAcc_base, miou_base, per_cls_base = evaluate_on_dataset(
        seg, configs[args.dataset], args.dataset, out_dir, use_dsm=False)
    seg.cleanup()
    results['baseline'] = {'aAcc': aAcc_base, 'mIoU': miou_base, 'per_class': per_cls_base}
    print(f"\n  Baseline: aAcc={aAcc_base:.1f}%, mIoU={miou_base:.1f}%")
    print(f"  Per-class: {[f'{v:.1f}' for v in per_cls_base]}")

    # === Approach A: DSM geometric prompts ===
    print(f"\n{'='*60}")
    print(f"Approach A — DSM Geometric Prompts — {args.dataset}")
    seg = DSMEnhancedSegmentor(
        classname_path=f'{PROJECT_ROOT}/configs/cls_{args.dataset}_noclutter.txt',
        prob_thd=0.1, bg_idx=0, confidence_threshold=0.4)
    out_dir = os.path.join(output_root, f'{args.dataset}_dsm_A')
    aAcc_a, miou_a, per_cls_a = evaluate_on_dataset(
        seg, configs[args.dataset], args.dataset, out_dir, use_dsm=True)
    seg.cleanup()
    results['dsm_A'] = {'aAcc': aAcc_a, 'mIoU': miou_a, 'per_class': per_cls_a}
    print(f"\n  DSM-A: aAcc={aAcc_a:.1f}%, mIoU={miou_a:.1f}%")
    print(f"  Per-class: {[f'{v:.1f}' for v in per_cls_a]}")
    print(f"  Δ vs baseline: mIoU {miou_a-miou_base:+.1f}%")

    # Save results
    json.dump(results, open(os.path.join(output_root, 'results.json'), 'w'), indent=2, default=str)
    print(f"\nResults saved to: {output_root}/results.json")

    # Summary table
    print(f"\n{'='*60}")
    print(f"Phase 4 Summary — {args.dataset}")
    print(f"  Baseline (no DSM):    {miou_base:.1f}%")
    print(f"  Approach A (DSM box): {miou_a:.1f}% (Δ={miou_a-miou_base:+.1f}%)")


if __name__ == '__main__':
    main()
