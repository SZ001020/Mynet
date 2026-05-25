"""Online random crop dataset for Plan6 Phase 1.5.

Tiles are cached in memory once, then each epoch samples fresh 256x256 crops.
This matches MFNet-style data diversity better than pre-extracting a fixed
window list.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image

from dataset_adapter import IGNORE_INDEX, _rgb_to_class


class OnlineCropDataset(torch.utils.data.Dataset):
    """Cache full tiles and sample random aligned RGB/DSM/label crops online."""

    def __init__(
        self,
        img_dir: str,
        gt_dir: str,
        tiles: list[str],
        img_suffix: str,
        gt_suffix: str,
        dsm_paths: dict[str, str],
        is_train: bool,
        crop_size: int = 256,
        epoch_steps: int = 1000,
        batch_size: int = 4,
    ):
        self.is_train = is_train
        self.crop_size = crop_size
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.images: list[np.ndarray] = []
        self.dsms: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.tile_names: list[str] = []

        for tile in tiles:
            ip = f"{img_dir}/{tile}{img_suffix}"
            gp = f"{gt_dir}/{tile}{gt_suffix}"
            dp = dsm_paths.get(tile)
            if not os.path.exists(ip) or not os.path.exists(gp):
                continue

            img = np.array(Image.open(ip).convert("RGB"))
            label = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            if dp and os.path.exists(dp):
                dsm = np.array(Image.open(dp)).astype(np.float32)
                dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
            else:
                dsm = np.zeros(img.shape[:2], dtype=np.float32)

            if img.shape[0] < crop_size or img.shape[1] < crop_size:
                pad_h = max(0, crop_size - img.shape[0])
                pad_w = max(0, crop_size - img.shape[1])
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                dsm = np.pad(dsm, ((0, pad_h), (0, pad_w)), mode="reflect")
                label = np.pad(
                    label,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=IGNORE_INDEX,
                )

            self.images.append(img)
            self.dsms.append(dsm)
            self.labels.append(label)
            self.tile_names.append(tile)

        if not self.images:
            raise RuntimeError("OnlineCropDataset found no valid tiles")

        print(
            f"  Cached {len(self.images)} tiles for {'train' if is_train else 'val'}; "
            f"online crops/epoch={len(self)}"
        )

    def __len__(self) -> int:
        if self.is_train:
            return self.batch_size * self.epoch_steps
        return len(self.images)

    def _random_crop(self, img: np.ndarray, dsm: np.ndarray, label: np.ndarray):
        h, w = img.shape[:2]
        y = random.randint(0, max(0, h - self.crop_size))
        x = random.randint(0, max(0, w - self.crop_size))
        y2, x2 = y + self.crop_size, x + self.crop_size
        return img[y:y2, x:x2], dsm[y:y2, x:x2], label[y:y2, x:x2]

    def _center_crop(self, img: np.ndarray, dsm: np.ndarray, label: np.ndarray):
        h, w = img.shape[:2]
        y = max(0, (h - self.crop_size) // 2)
        x = max(0, (w - self.crop_size) // 2)
        y2, x2 = y + self.crop_size, x + self.crop_size
        return img[y:y2, x:x2], dsm[y:y2, x:x2], label[y:y2, x:x2]

    def __getitem__(self, idx: int):
        tile_idx = random.randrange(len(self.images)) if self.is_train else idx
        img = self.images[tile_idx]
        dsm = self.dsms[tile_idx]
        label = self.labels[tile_idx]

        if self.is_train:
            img, dsm, label = self._random_crop(img, dsm, label)
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                dsm = np.fliplr(dsm).copy()
                label = np.fliplr(label).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                dsm = np.flipud(dsm).copy()
                label = np.flipud(label).copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k).copy()
                dsm = np.rot90(dsm, k).copy()
                label = np.rot90(label, k).copy()
        else:
            img, dsm, label = self._center_crop(img, dsm, label)

        return (
            torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(dsm.copy()).float(),
            torch.from_numpy(label.copy()).long(),
        )
