"""
遥感数据集 — per-class 二分类格式。
每张图 × 每类 = 一个 Sample(image, binary_gt, class_name).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List
import numpy as np
from PIL import Image


@dataclass
class Sample:
    image: np.ndarray          # RGB (H, W, 3)
    gt_mask: np.ndarray        # binary mask (H, W), 0 or 1
    dataset_name: str
    tile_name: str
    class_name: str
    class_index: int


# 5 类 (no clutter), ISPRS label mapping: 1=road, 2=building, 3=grass, 4=tree, 5=car
CLASSES = [
    ('road', 1),
    ('building', 2),
    ('grass', 3),
    ('tree', 4),
    ('car', 5),
]

# Text prompts (Phase 1 simple-word optimal)
TEXT_PROMPTS = {name: name for name, _ in CLASSES}


def get_vaihingen_tiles():
    return [f'top_mosaic_09cm_area{i}' for i in [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]]


def get_potsdam_tiles():
    tiles = []
    for t in ['2','3','4','5']:
        for n in ['10','11','12']:
            tiles.append(f'top_potsdam_{t}_{n}')
    for t in ['6','7']:
        for n in ['7','8','9','10','11','12']:
            tiles.append(f'top_potsdam_{t}_{n}')
    return tiles


def load_rs_samples(dataset_name: str, max_samples: Optional[int] = None) -> Iterator[Sample]:
    """Load per-class binary samples for a remote sensing dataset."""
    if dataset_name == 'vaihingen':
        tiles = get_vaihingen_tiles()
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_index'
        img_suf, gt_suf = '.tif', '.png'
    elif dataset_name == 'potsdam':
        tiles = get_potsdam_tiles()
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/labels_index'
        img_suf, gt_suf = '_RGB.tif', '.png'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    count = 0
    for tile in tiles:
        img_path = os.path.join(img_dir, f'{tile}{img_suf}')
        gt_path = os.path.join(gt_dir, f'{tile}{gt_suf}')
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(gt_path))

        # Pre-resize large images
        max_edge = 2000
        h, w = image.shape[:2]
        if max(h, w) > max_edge:
            ratio = max_edge / max(h, w)
            new_size = (int(w * ratio), int(h * ratio))
            image = np.array(Image.fromarray(image).resize(new_size, Image.BILINEAR))
            gt = np.array(Image.fromarray(gt).resize(new_size, Image.NEAREST))

        for class_name, label_val in CLASSES:
            binary_mask = (gt == label_val).astype(np.uint8)
            yield Sample(
                image=image, gt_mask=binary_mask,
                dataset_name=dataset_name, tile_name=tile,
                class_name=class_name, class_index=label_val - 1,
            )
            count += 1
            if max_samples and count >= max_samples:
                return
