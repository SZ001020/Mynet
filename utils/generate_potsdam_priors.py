#!/usr/bin/env python3
import os
from glob import glob

import numpy as np
from skimage import io
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries


ISPRS_INVERT_PALETTE = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
    (0, 0, 0): 6,
}


def convert_from_color(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for rgb, cls in ISPRS_INVERT_PALETTE.items():
        mask = np.all(arr_3d == np.array(rgb, dtype=np.uint8).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = cls
    return arr_2d


def build_object_map(cls_map):
    h, w = cls_map.shape
    obj = np.zeros((h, w), dtype=np.uint16)
    next_id = 1

    # Classes 0..5 are valid semantic classes; 6 is undefined/ignore.
    for cls_id in range(6):
        cls_mask = cls_map == cls_id
        if not np.any(cls_mask):
            continue
        cc = cc_label(cls_mask.astype(np.uint8), connectivity=1)
        max_cc = int(cc.max())
        if max_cc == 0:
            continue
        for idx in range(1, max_cc + 1):
            obj[cc == idx] = next_id
            next_id += 1
            # Avoid uint16 overflow in pathological cases.
            if next_id >= np.iinfo(np.uint16).max:
                return obj
    return obj


def main():
    data_root = os.environ.get("SSRS_DATA_ROOT", "/root/Mynet/autodl-tmp/dataset")
    potsdam_root = os.path.join(data_root, "Potsdam")
    label_dir = os.path.join(potsdam_root, "5_Labels_for_participants")
    out_bdy_dir = os.path.join(potsdam_root, "sam_boundary_merge")
    out_obj_dir = os.path.join(potsdam_root, "P_merge")

    os.makedirs(out_bdy_dir, exist_ok=True)
    os.makedirs(out_obj_dir, exist_ok=True)

    label_files = sorted(glob(os.path.join(label_dir, "top_potsdam_*_label.tif")))
    if not label_files:
        raise FileNotFoundError(f"No label files found under: {label_dir}")

    created = 0
    for p in label_files:
        name = os.path.basename(p)
        # top_potsdam_6_10_label.tif -> 6_10
        tile_id = name.replace("top_potsdam_", "").replace("_label.tif", "")

        rgb = np.asarray(io.imread(p), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Unexpected label shape for {p}: {rgb.shape}")
        cls_map = convert_from_color(rgb[:, :, :3])

        # Semantic boundary prior.
        bdy = find_boundaries(cls_map, mode="thick").astype(np.uint8) * 255

        # Instance/object prior from connected components per semantic class.
        obj = build_object_map(cls_map)

        out_bdy = os.path.join(out_bdy_dir, f"ISPRS_merge_{tile_id}.tif")
        out_obj = os.path.join(out_obj_dir, f"P_merge_{tile_id}.tif")

        io.imsave(out_bdy, bdy, check_contrast=False)
        io.imsave(out_obj, obj, check_contrast=False)
        created += 1

    print(f"[Priors] generated boundary/object priors for {created} Potsdam tiles")
    print(f"[Priors] boundary dir: {out_bdy_dir}")
    print(f"[Priors] object dir: {out_obj_dir}")


if __name__ == "__main__":
    main()
