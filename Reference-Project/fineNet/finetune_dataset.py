"""
Phase 3: 训练数据集加载器

使用 Vaihingen 和 Potsdam 的 GT 数据创建训练/验证集。
5 类 (road, building, grass, tree, car)，clutter 映射为 ignore_index。
图像随机裁剪到 512×512，配合数据增强。
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


CLASS_MAPPING = {
    # ISPRS label → 5-class index
    # Original: 1=road, 2=building, 3=grass, 4=tree, 5=car, 6=clutter
    1: 0,  # road
    2: 1,  # building
    3: 2,  # grass
    4: 3,  # tree
    5: 4,  # car
    6: 255,  # clutter → ignore
}

NUM_CLASSES = 5
IGNORE_INDEX = 255


def get_vaihingen_splits():
    """返回 Vaihingen 训练/验证 tile 列表."""
    all_tiles = [
        'top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
        'top_mosaic_09cm_area7', 'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13',
        'top_mosaic_09cm_area15', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area21',
        'top_mosaic_09cm_area23', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
        'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34',
        'top_mosaic_09cm_area37',
    ]
    # 12 train, 4 val
    train = all_tiles[:12]
    val = all_tiles[12:]
    return train, val


def get_potsdam_splits():
    """返回 Potsdam 训练/验证 tile 列表."""
    all_tiles = []
    for t in ['2', '3', '4', '5', '6', '7']:
        for n in ['10', '11', '12']:
            all_tiles.append(f'top_potsdam_{t}_{n}')
    # Also include 6_7, 6_8, 6_9, 7_7, 7_8, 7_9
    for t in ['6', '7']:
        for n in ['7', '8', '9']:
            all_tiles.append(f'top_potsdam_{t}_{n}')
    # 18 train, 6 val
    train = all_tiles[:18]
    val = all_tiles[18:]
    return train, val


class RSFineTuneDataset(Dataset):
    """遥感语义分割训练数据集。

    Args:
        tile_list: 瓦片名称列表
        data_root: 数据集根目录
        dataset_name: 'vaihingen' or 'potsdam'
        crop_size: 随机裁剪尺寸
        is_train: 是否使用数据增强
    """

    def __init__(self, tile_list, data_root, dataset_name,
                 crop_size=512, is_train=True):
        self.tile_list = tile_list
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.is_train = is_train

        # Build image and label paths
        self.samples = []
        for tile in tile_list:
            if dataset_name == 'vaihingen':
                img_path = os.path.join(data_root, 'top', f'{tile}.tif')
                gt_path = os.path.join(data_root, 'gts_index', f'{tile}.png')
            else:  # potsdam
                img_path = os.path.join(data_root, '2_Ortho_RGB', f'{tile}_RGB.tif')
                gt_path = os.path.join(data_root, 'labels_index', f'{tile}.png')

            if os.path.exists(img_path) and os.path.exists(gt_path):
                self.samples.append((img_path, gt_path))

        print(f"  {dataset_name}: {len(self.samples)} tiles loaded ({'train' if is_train else 'val'})")

    def __len__(self):
        return len(self.samples) * (80 if self.is_train else 10)

    def __getitem__(self, idx):
        tile_idx = idx % len(self.samples)
        img_path, gt_path = self.samples[tile_idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(gt_path))

        # Remap labels: 1-6 → 0-4 (clutter→255)
        label_remapped = np.full_like(label, IGNORE_INDEX, dtype=np.int64)
        for orig, new in CLASS_MAPPING.items():
            label_remapped[label == orig] = new

        # Random crop
        if self.is_train:
            h, w = image.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                image = image[y:y+self.crop_size, x:x+self.crop_size]
                label_remapped = label_remapped[y:y+self.crop_size, x:x+self.crop_size]

            # Augmentations
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label_remapped = np.fliplr(label_remapped).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                label_remapped = np.flipud(label_remapped).copy()
            # Random rotation 0/90/180/270
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                label_remapped = np.rot90(label_remapped, k).copy()

        # To tensor, normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_remapped = torch.from_numpy(label_remapped).long()

        return image, label_remapped


def create_dataloaders(batch_size=4, num_workers=4, crop_size=512, dataset_name=None):
    """Create train/val dataloaders. If dataset_name is specified, use only that dataset."""

    if dataset_name in (None, 'vaihingen'):
        v_train_tiles, v_val_tiles = get_vaihingen_splits()
    if dataset_name in (None, 'potsdam'):
        p_train_tiles, p_val_tiles = get_potsdam_splits()

    if dataset_name == 'vaihingen':
        train_dataset = RSFineTuneDataset(v_train_tiles, '/root/Mynet/autodl-tmp/dataset/Vaihingen',
                                          'vaihingen', crop_size=crop_size, is_train=True)
        v_val_dataset = RSFineTuneDataset(v_val_tiles, '/root/Mynet/autodl-tmp/dataset/Vaihingen',
                                           'vaihingen', crop_size=crop_size, is_train=False)
        p_val_dataset = RSFineTuneDataset(v_val_tiles, '/root/Mynet/autodl-tmp/dataset/Vaihingen',
                                           'vaihingen', crop_size=crop_size, is_train=False)
    elif dataset_name == 'potsdam':
        train_dataset = RSFineTuneDataset(p_train_tiles, '/root/Mynet/autodl-tmp/dataset/Potsdam',
                                          'potsdam', crop_size=crop_size, is_train=True)
        v_val_dataset = RSFineTuneDataset(p_val_tiles, '/root/Mynet/autodl-tmp/dataset/Potsdam',
                                           'potsdam', crop_size=crop_size, is_train=False)
        p_val_dataset = RSFineTuneDataset(p_val_tiles, '/root/Mynet/autodl-tmp/dataset/Potsdam',
                                           'potsdam', crop_size=crop_size, is_train=False)
    else:
        train_dataset = torch.utils.data.ConcatDataset([
            RSFineTuneDataset(v_train_tiles, '/root/Mynet/autodl-tmp/dataset/Vaihingen',
                              'vaihingen', crop_size=crop_size, is_train=True),
            RSFineTuneDataset(p_train_tiles, '/root/Mynet/autodl-tmp/dataset/Potsdam',
                              'potsdam', crop_size=crop_size, is_train=True),
        ])
        v_val_dataset = RSFineTuneDataset(v_val_tiles, '/root/Mynet/autodl-tmp/dataset/Vaihingen',
                                           'vaihingen', crop_size=crop_size, is_train=False)
        p_val_dataset = RSFineTuneDataset(p_val_tiles, '/root/Mynet/autodl-tmp/dataset/Potsdam',
                                           'potsdam', crop_size=crop_size, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    v_val_loader = torch.utils.data.DataLoader(
        v_val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    p_val_loader = torch.utils.data.DataLoader(
        p_val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, v_val_loader, p_val_loader
