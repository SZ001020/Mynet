#!/usr/bin/env python3
import os
import sys
import numpy as np
from skimage import io


def build_paths(data_root, dataset, tile_id):
    if dataset == "Vaihingen":
        bdy = os.path.join(data_root, "Vaihingen", "sam_boundary_merge", f"ISPRS_merge_{tile_id}.tif")
        obj = os.path.join(data_root, "Vaihingen", "V_merge", f"V_merge_{tile_id}.tif")
    elif dataset == "Potsdam":
        bdy = os.path.join(data_root, "Potsdam", "sam_boundary_merge", f"ISPRS_merge_{tile_id}.tif")
        obj = os.path.join(data_root, "Potsdam", "P_merge", f"P_merge_{tile_id}.tif")
    else:
        raise ValueError(f"Unsupported dataset for checker: {dataset}")
    return bdy, obj


def default_train_ids(dataset):
    if dataset == "Vaihingen":
        return ["1", "3", "23", "26", "7", "11", "13", "28", "17", "32", "34", "37"]
    if dataset == "Potsdam":
        return [
            "6_10", "7_10", "2_12", "3_11", "2_10", "7_8", "5_10", "3_12", "5_12",
            "7_11", "7_9", "6_9", "7_7", "4_12", "6_8", "6_12", "6_7", "4_11",
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    data_root = os.environ.get("SSRS_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "autodl-tmp", "dataset"))
    dataset = os.environ.get("SSRS_DATASET", "Vaihingen")
    sample_limit = int(os.environ.get("SSRS_PRIOR_CHECK_SAMPLES", "0"))

    ids = default_train_ids(dataset)
    if sample_limit > 0:
        ids = ids[:sample_limit]

    missing_bdy, missing_obj = [], []
    bdy_ratios, obj_ratios, obj_counts = [], [], []

    for tile_id in ids:
        bdy_path, obj_path = build_paths(data_root, dataset, tile_id)
        if not os.path.isfile(bdy_path):
            missing_bdy.append(bdy_path)
        if not os.path.isfile(obj_path):
            missing_obj.append(obj_path)
        if (not os.path.isfile(bdy_path)) or (not os.path.isfile(obj_path)):
            continue

        bdy = np.asarray(io.imread(bdy_path))
        if bdy.ndim == 3:
            bdy = bdy[:, :, 0]
        bdy_bin = (bdy > 0).astype(np.uint8)

        obj = np.asarray(io.imread(obj_path))
        if obj.ndim == 3:
            obj = obj[:, :, 0]
        obj = obj.astype(np.int64)

        bdy_ratios.append(float(np.mean(bdy_bin)))
        obj_ratios.append(float(np.mean(obj > 0)))
        obj_counts.append(int(len(np.unique(obj[obj > 0]))))

    print(f"[PriorCheck] dataset={dataset}, tiles={len(ids)}")
    print(f"[PriorCheck] missing_bdy={len(missing_bdy)}, missing_obj={len(missing_obj)}")
    if missing_bdy:
        print(f"[PriorCheck] missing boundary examples: {missing_bdy[:3]}")
    if missing_obj:
        print(f"[PriorCheck] missing object examples: {missing_obj[:3]}")

    if not bdy_ratios:
        print("[PriorCheck][FAIL] no valid boundary/object pairs found")
        sys.exit(2)

    bdy_mean = float(np.mean(bdy_ratios))
    obj_mean = float(np.mean(obj_ratios))
    obj_cnt_mean = float(np.mean(obj_counts))

    print(
        "[PriorCheck] boundary ratio mean/min/max = "
        f"{bdy_mean:.6f}/{np.min(bdy_ratios):.6f}/{np.max(bdy_ratios):.6f}"
    )
    print(
        "[PriorCheck] object ratio   mean/min/max = "
        f"{obj_mean:.6f}/{np.min(obj_ratios):.6f}/{np.max(obj_ratios):.6f}"
    )
    print(f"[PriorCheck] object instances per tile mean = {obj_cnt_mean:.2f}")

    warnings = []
    if bdy_mean < 0.003:
        warnings.append("boundary ratio too low")
    if obj_mean < 0.02:
        warnings.append("object ratio too low")
    if obj_cnt_mean < 5:
        warnings.append("object instance count too low")

    if warnings:
        print("[PriorCheck][WARN] " + ", ".join(warnings))
        sys.exit(1)

    print("[PriorCheck][PASS] prior statistics look reasonable")


if __name__ == "__main__":
    main()
