#!/usr/bin/env python3
"""Generate boundary maps and object instance maps from GT labels.

Boundary: morphological gradient (dilation - erosion)
Object: connected components per class, stored as unique instance IDs
"""

import os, numpy as np
from PIL import Image
from scipy import ndimage

from dataset_adapter import (_rgb_to_class, VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                             POTSDAM_TRAIN, POTSDAM_VAL)

OUT_DIR = '/root/autodl-tmp/dataset/Vaihingen/boundary_object'
os.makedirs(OUT_DIR, exist_ok=True)

for tag, tiles in [('train', VAIHINGEN_TRAIN), ('val', VAIHINGEN_VAL)]:
    for tile in tiles:
        gp = f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{tile}.tif'
        if not os.path.exists(gp):
            continue
        gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))
        h, w = gt.shape

        # Boundary: per-class morphological gradient, then union
        boundary = np.zeros((h, w), dtype=np.uint8)
        for c in range(5):  # 5 foreground classes
            mask = (gt == c).astype(np.uint8)
            if mask.sum() == 0:
                continue
            dilated = ndimage.binary_dilation(mask, iterations=1)
            eroded = ndimage.binary_erosion(mask, iterations=1)
            boundary |= (dilated ^ eroded)  # XOR = boundary

        # Object instances: connected components per class
        object_map = np.zeros((h, w), dtype=np.int32)
        instance_id = 1
        for c in range(5):
            mask = (gt == c)
            labeled, n = ndimage.label(mask)
            for i in range(1, n + 1):
                object_map[labeled == i] = instance_id
                instance_id += 1

        np.savez_compressed(f'{OUT_DIR}/{tile}.npz',
                            boundary=boundary, objects=object_map)
        print(f'  {tile}: {instance_id-1} instances, boundary={boundary.sum()}px')

print(f'Done! Output: {OUT_DIR}')
