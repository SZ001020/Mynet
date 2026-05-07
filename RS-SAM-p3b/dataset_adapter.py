"""
Plan3 Route A: 训练数据集加载器 (MFNet split)

与 MFNet 完全一致的 train/val 划分:
- Vaihingen: 12 train (MFNet train_ids) / 4 val (MFNet test_ids)
- Potsdam: 18 train (MFNet train_ids) / 6 val (MFNet test_ids)
- 随机 crop 512×512, flip/rotate augmentation
- 5 类: road(0), building(1), grass(2), tree(3), car(4), clutter→ignore(255)
- 标签来源: gts_for_participants (0-based, 与 MFNet 一致)
"""

import os, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# gts_for_participants are RGB palette images. Convert RGB → class index.
# ISPRS palette: 0=road(255,255,255), 1=building(0,0,255), 2=lowVeg(0,255,255),
#                 3=tree(0,255,0), 4=car(255,255,0), 5=clutter(255,0,0)
RGB_TO_CLASS = {
    (255, 255, 255): 0,  # road
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # grass (low vegetation)
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 255,    # clutter → ignore
}
NUM_CLASSES = 5
IGNORE_INDEX = 255


# ── DSM directory mapping ──────────────────────────────────

DSM_PATTERNS = {
    'vaihingen': ('/root/autodl-tmp/dataset/Vaihingen/dsm',
                  'dsm_09cm_matching_area{}.tif',
                  lambda t: t.replace('top_mosaic_09cm_area', '')),
    'potsdam': ('/root/autodl-tmp/dataset/Potsdam/1_DSM',
                'dsm_potsdam_{}.tif',
                lambda t: t.replace('top_potsdam_', '')),
}


def _rgb_to_class(label_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB palette label (H,W,3) to class index (H,W)."""
    h, w, _ = label_rgb.shape
    label_idx = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
    for rgb, cls in RGB_TO_CLASS.items():
        mask = np.all(label_rgb == rgb, axis=-1)
        label_idx[mask] = cls
    return label_idx

# ── MFNet splits (exact match with MFNet/utils.py) ──────────

VAIHINGEN_TRAIN = [f'top_mosaic_09cm_area{i}' for i in
                   [1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37]]
VAIHINGEN_VAL = [f'top_mosaic_09cm_area{i}' for i in [5, 21, 15, 30]]

POTSDAM_TRAIN = [f'top_potsdam_{tid}' for tid in
    ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12',
     '5_12', '7_11', '7_9', '6_9', '7_7', '4_12', '6_8', '6_12', '6_7', '4_11']]
POTSDAM_VAL = [f'top_potsdam_{tid}' for tid in
    ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']]


class ISPRSTrainDataset(Dataset):
    """Per-tile ISPRS dataset with random cropping and augmentation.

    Caches all tiles in memory (like MFNet's CACHE=True) to avoid
    repeated TIF I/O. Potsdam tiles at 6000² × 18 ≈ 1.9GB RAM — fine with 754GB.
    """

    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                 crop_size=512, is_train=True, n_crops=80):
        self.crop_size = crop_size
        self.is_train = is_train
        self.n_crops = n_crops if is_train else 10

        # Pre-load all tiles into memory (one-time TIF I/O)
        self.images = []
        self.labels = []
        for t in tiles:
            ip = os.path.join(img_dir, f'{t}{img_suffix}')
            gp = os.path.join(gt_dir, f'{t}{gt_suffix}')
            if not os.path.exists(ip):
                continue
            img = np.array(Image.open(ip).convert('RGB'))
            label_rgb = np.array(Image.open(gp).convert('RGB'))
            label_idx = _rgb_to_class(label_rgb)
            self.images.append(img)
            self.labels.append(label_idx)

        print(f"  {len(self.images)} tiles loaded to memory ({'train' if is_train else 'val'})")

    def __len__(self):
        return len(self.images) * self.n_crops

    def __getitem__(self, idx):
        tile_idx = idx % len(self.images)
        image = self.images[tile_idx]
        label_remapped = self.labels[tile_idx]

        if self.is_train:
            h, w = image.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                image = image[y:y+self.crop_size, x:x+self.crop_size]
                label_remapped = label_remapped[y:y+self.crop_size, x:x+self.crop_size]
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label_remapped = np.fliplr(label_remapped).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                label_remapped = np.flipud(label_remapped).copy()
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                label_remapped = np.rot90(label_remapped, k).copy()

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_remapped = torch.from_numpy(label_remapped).long()
        return image, label_remapped


class ISPRSTrainDatasetDSM(ISPRSTrainDataset):
    """ISPRS dataset with DSM loading alongside RGB."""

    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                 dsm_dir, dsm_pattern, dsm_stem_fn,
                 crop_size=512, is_train=True, n_crops=80):
        super().__init__(img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                         crop_size, is_train, n_crops)
        # Load DSMs alongside images
        self.dsms = []
        for t in tiles:
            stem = dsm_stem_fn(t)
            dp = os.path.join(dsm_dir, dsm_pattern.format(stem))
            if os.path.exists(dp):
                dsm_img = np.array(Image.open(dp)).astype(np.float32)
                # Normalize to [0, 1] using per-tile stats
                dsm_img = (dsm_img - dsm_img.min()) / max(dsm_img.max() - dsm_img.min(), 1e-8)
                self.dsms.append(dsm_img)
            else:
                # Fallback: zeros
                self.dsms.append(np.zeros_like(self.images[-1][:, :, 0], dtype=np.float32))
        print(f"  {len(self.dsms)} DSMs loaded to memory")

    def __getitem__(self, idx):
        tile_idx = idx % len(self.images)
        image = self.images[tile_idx]
        label_remapped = self.labels[tile_idx]
        dsm = self.dsms[tile_idx]

        h, w = image.shape[:2]
        if self.is_train:
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                image = image[y:y+self.crop_size, x:x+self.crop_size]
                label_remapped = label_remapped[y:y+self.crop_size, x:x+self.crop_size]
                dsm = dsm[y:y+self.crop_size, x:x+self.crop_size]
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label_remapped = np.fliplr(label_remapped).copy()
                dsm = np.fliplr(dsm).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                label_remapped = np.flipud(label_remapped).copy()
                dsm = np.flipud(dsm).copy()
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                label_remapped = np.rot90(label_remapped, k).copy()
                dsm = np.rot90(dsm, k).copy()

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_remapped = torch.from_numpy(label_remapped).long()
        dsm_t = torch.from_numpy(dsm).float()
        return image, dsm_t, label_remapped


def create_dataloaders(batch_size=4, crop_size=512, dataset_name=None, num_workers=2):
    """Create train/val dataloaders for Vaihingen or Potsdam.

    Uses MFNet's exact train/val splits and gts_for_participants labels.
    """
    VAIH = '/root/autodl-tmp/dataset/Vaihingen'
    POTS = '/root/autodl-tmp/dataset/Potsdam'

    if dataset_name == 'vaihingen':
        train_ds = ISPRSTrainDataset(
            f'{VAIH}/top', f'{VAIH}/gts_for_participants',
            VAIHINGEN_TRAIN, '.tif', '.tif', crop_size, True)
        v_val_ds = ISPRSTrainDataset(
            f'{VAIH}/top', f'{VAIH}/gts_for_participants',
            VAIHINGEN_VAL, '.tif', '.tif', crop_size, False)
        # For potsdam val when training only vaihingen, use vaihingen val as placeholder
        p_val_ds = v_val_ds
    elif dataset_name == 'potsdam':
        train_ds = ISPRSTrainDataset(
            f'{POTS}/2_Ortho_RGB', f'{POTS}/5_Labels_for_participants',
            POTSDAM_TRAIN, '_RGB.tif', '_label.tif', crop_size, True)
        v_val_ds = ISPRSTrainDataset(
            f'{POTS}/2_Ortho_RGB', f'{POTS}/5_Labels_for_participants',
            POTSDAM_VAL, '_RGB.tif', '_label.tif', crop_size, False)
        p_val_ds = v_val_ds
    else:
        # Both datasets combined (rarely used, kept for compatibility)
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([
            ISPRSTrainDataset(f'{VAIH}/top', f'{VAIH}/gts_for_participants',
                              VAIHINGEN_TRAIN, '.tif', '.tif', crop_size, True),
            ISPRSTrainDataset(f'{POTS}/2_Ortho_RGB', f'{POTS}/5_Labels_for_participants',
                              POTSDAM_TRAIN, '_RGB.tif', '_label.tif', crop_size, True),
        ])
        v_val_ds = ISPRSTrainDataset(f'{VAIH}/top', f'{VAIH}/gts_for_participants',
                                     VAIHINGEN_VAL, '.tif', '.tif', crop_size, False)
        p_val_ds = ISPRSTrainDataset(f'{POTS}/2_Ortho_RGB', f'{POTS}/5_Labels_for_participants',
                                     POTSDAM_VAL, '_RGB.tif', '_label.tif', crop_size, False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    v_val_loader = DataLoader(v_val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    p_val_loader = DataLoader(p_val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, v_val_loader, p_val_loader
