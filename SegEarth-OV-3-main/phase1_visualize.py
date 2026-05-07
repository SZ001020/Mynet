#!/usr/bin/env python3
"""
Phase 1 可视化: 对代表性样本生成各 head 配置的预测图对比。
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from mmengine.config import Config
from mmengine.runner import Runner
import custom_datasets  # noqa
import segearthov3_segmentor  # noqa

# ============================================================
# 配置
# ============================================================

DATASET_CONFIGS = {
    'vaihingen': {
        'config': './configs/cfg_vaihingen.py',
        'palette': [
            [255, 255, 255],  # 0 road (impervious) - white
            [0, 0, 255],      # 1 building - blue
            [0, 255, 255],    # 2 grass (low_veg) - cyan
            [0, 255, 0],      # 3 tree - green
            [255, 255, 0],    # 4 car - yellow
            [255, 0, 0],      # 5 clutter - red
        ],
        'class_names': ['road', 'building', 'grass', 'tree', 'car', 'clutter'],
        'sample_indices': [0, 3, 7],  # Use first few validation images
    },
    'potsdam': {
        'config': './configs/cfg_potsdam_fast.py',
        'palette': [
            [255, 255, 255],  # road
            [0, 0, 255],      # building
            [0, 255, 255],    # grass
            [0, 255, 0],      # tree
            [255, 255, 0],    # car
            [255, 0, 0],      # clutter
        ],
        'class_names': ['road', 'building', 'grass', 'tree', 'car', 'clutter'],
        'sample_indices': [0, 2, 5],
    },
}

HEAD_CONFIGS = [
    ('Instance',              True,  False, False),
    ('Instance+Presence',     True,  False, True),
    ('Semantic',              False, True,  False),
    ('Semantic+Presence',     False, True,  True),
    ('Dual-Head',             True,  True,  False),
    ('Dual-Head+Presence',    True,  True,  True),
]

OUTPUT_DIR = '/root/Mynet/autodl-tmp/runs/phase1_visualizations'


def pred_to_color(pred_mask, palette):
    """Convert class index prediction to RGB color image."""
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        color_img[pred_mask == cls_idx] = color
    return color_img


def run_inference(cfg_path, image_path, use_td, use_ss, use_ps):
    """对单张图像运行 SAM 3 推理，返回类别预测 (H, W) numpy int."""
    # Build config
    cfg = Config.fromfile(cfg_path)
    cfg.model.use_transformer_decoder = use_td
    cfg.model.use_sem_seg = use_ss
    cfg.model.use_presence_score = use_ps

    # Build model manually for single-image inference
    from segearthov3_segmentor import SegEarthOV3Segmentation, get_cls_idx

    model_kwargs = {}
    for k in ['classname_path', 'prob_thd', 'bg_idx', 'confidence_threshold',
              'slide_stride', 'slide_crop']:
        if k in cfg.model:
            model_kwargs[k] = cfg.model[k]
    for k in ['use_sem_seg', 'use_presence_score', 'use_transformer_decoder']:
        model_kwargs[k] = cfg.model.get(k, True)

    model = SegEarthOV3Segmentation(**model_kwargs)

    # Run inference
    image = Image.open(image_path).convert('RGB')
    max_edge = 2000
    if max(image.size) > max_edge:
        ratio = max_edge / max(image.size)
        image = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)),
            Image.BILINEAR)

    seg_logits = model._inference_single_view(image)

    if model.num_cls != model.num_queries:
        seg_logits = seg_logits.unsqueeze(0)
        cls_index = torch.nn.functional.one_hot(model.query_idx)
        cls_index = cls_index.T.view(model.num_cls, len(model.query_idx), 1, 1)
        seg_logits = (seg_logits * cls_index.to(seg_logits.device)).max(1)[0]

    seg_pred = seg_logits.argmax(0).cpu().numpy()
    max_vals = seg_logits.max(0)[0].cpu().numpy()
    seg_pred[max_vals < model.prob_thd] = model.bg_idx

    # Clean up
    del model
    torch.cuda.empty_cache()

    return seg_pred


def load_gt(image_path, dataset_key):
    """Load ground truth mask for comparison."""
    if dataset_key == 'vaihingen':
        gt_path = image_path.replace('/top/', '/gts_index/')
        gt_path = gt_path.replace('.tif', '.png')
    elif dataset_key == 'potsdam':
        gt_path = image_path.replace('2_Ortho_RGB', 'labels_index')
        gt_path = gt_path.replace('_RGB.tif', '.png')
    else:
        return None

    if os.path.exists(gt_path):
        gt = np.array(Image.open(gt_path))
        return gt
    return None


def get_validation_images(dataset_key):
    """获取验证集图片路径列表"""
    if dataset_key == 'vaihingen':
        val_file = '/root/Mynet/SegEarth-OV-3-main/configs/vaihingen_val.txt'
        img_dir = '/root/Mynet/autodl-tmp/dataset/Vaihingen/top'
        suffix = '.tif'
    elif dataset_key == 'potsdam':
        val_file = '/root/Mynet/SegEarth-OV-3-main/configs/potsdam_val.txt'
        img_dir = '/root/Mynet/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        suffix = '_RGB.tif'

    with open(val_file, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]

    paths = [os.path.join(img_dir, f'{tile}{suffix}') for tile in tiles]
    return [p for p in paths if os.path.exists(p)]


def create_comparison_figure(dataset_key, image_path, predictions, gt, palette, class_names, config_labels):
    """创建对比图: RGB | GT | 6个预测 排列成 2行×4列"""
    rgb = np.array(Image.open(image_path).convert('RGB'))
    # Resize to display size
    max_display = 600
    ratio = max_display / max(rgb.shape[:2])
    rgb_disp = np.array(Image.fromarray(rgb).resize(
        (int(rgb.shape[1]*ratio), int(rgb.shape[0]*ratio)), Image.LANCZOS))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # RGB
    axes[0].imshow(rgb_disp)
    axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # GT
    if gt is not None:
        gt_color = pred_to_color(gt, palette)
        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 6 predictions
    for i, (label, pred) in enumerate(zip(config_labels, predictions)):
        pred_color = pred_to_color(pred, palette)
        axes[i + 2].imshow(pred_color)
        axes[i + 2].set_title(label, fontsize=11, fontweight='bold')
        axes[i + 2].axis('off')

    # Legend
    legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n)
                      for c, n in zip(palette, class_names)]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=len(class_names), fontsize=10, frameon=False)

    fig.suptitle(f'{dataset_key.upper()} — Head Configuration Comparison',
                 fontsize=14, fontweight='bold', y=0.98)

    # Metrics annotation if available
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_key, ds_info in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_key}...")

        image_paths = get_validation_images(dataset_key)
        palette = ds_info['palette']
        class_names = ds_info['class_names']
        config_labels = [h[0] for h in HEAD_CONFIGS]

        for idx in ds_info['sample_indices']:
            if idx >= len(image_paths):
                continue
            img_path = image_paths[idx]
            tile_name = os.path.splitext(os.path.basename(img_path))[0]
            if tile_name.endswith('_RGB'):
                tile_name = tile_name[:-4]

            print(f"  Image {idx}: {tile_name}")

            # Load GT
            gt = load_gt(img_path, dataset_key)

            # Run all configs
            predictions = []
            for label, use_td, use_ss, use_ps in HEAD_CONFIGS:
                print(f"    Running {label}...")
                pred = run_inference(ds_info['config'], img_path, use_td, use_ss, use_ps)
                predictions.append(pred)
                # Save individual prediction
                pred_color = pred_to_color(pred, palette)
                out_name = f'{dataset_key}_{tile_name}_{label.replace("+", "-").replace(" ", "_")}.png'
                Image.fromarray(pred_color).save(os.path.join(OUTPUT_DIR, out_name))

            # Create comparison figure
            fig = create_comparison_figure(
                dataset_key, img_path, predictions, gt,
                palette, class_names, config_labels)
            fig_path = os.path.join(OUTPUT_DIR, f'comparison_{dataset_key}_{tile_name}.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path}")

    # Summary text
    summary_path = os.path.join(OUTPUT_DIR, 'visualization_notes.md')
    with open(summary_path, 'w') as f:
        f.write(f"""# Phase 1 推理图对比分析

> 生成时间: {datetime.now().isoformat()}
> 输出目录: {OUTPUT_DIR}

## 对比配置

| 缩写 | 含义 |
|------|------|
| Instance | Transformer Decoder 仅实例 mask |
| Instance+Pres | + Presence Score 过滤 |
| Semantic | Segmentation Head 仅语义 mask |
| Semantic+Pres | + Presence Score 过滤 |
| Dual-Head | 双头融合 (element-wise max) |
| Dual-Head+Pres | 双头 + Presence (默认配置) |

## 每个样本的输出文件

- `comparison_{{dataset}}_{{tile}}.png` — 8 图对比 (RGB + GT + 6 配置)
- `{{dataset}}_{{tile}}_{{config}}.png` — 单张配置预测图

## 观察要点

1. **Semantic vs Instance**: 注意密集区域（道路/草地）和离散物体（建筑/车辆）在两个 head 下的差异
2. **Presence 效果**: 对比 ±Presence 配置，观察误检的抑制情况
3. **Dual-Head 融合**: 观察双头融合是否能同时保留实例边界和语义覆盖
4. **Clutter 问题**: 注意 clutter（红色）类的分布——这是 SAM 3 最弱的类别
""")
    print(f"\nSummary: {summary_path}")
    print(f"All outputs: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
