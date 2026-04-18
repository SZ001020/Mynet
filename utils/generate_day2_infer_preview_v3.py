#!/usr/bin/env python3
import argparse
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from tqdm.auto import tqdm


ISPRS_PALETTE_CLS2RGB = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0),
}
ISPRS_PALETTE_RGB2CLS = {v: k for k, v in ISPRS_PALETTE_CLS2RGB.items()}


@dataclass
class BestMetric:
    epoch: int
    miou: float


def parse_args():
    p = argparse.ArgumentParser(description="Generate day2 inference preview images with artifact-safe resizing.")
    p.add_argument("--old-run", required=True, help="old day2 run dir")
    p.add_argument("--new-run", required=True, help="new day2 run dir")
    p.add_argument("--data-root", default="/root/Mynet/autodl-tmp/dataset")
    p.add_argument("--source", default="Vaihingen")
    p.add_argument("--target", default="Potsdam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--old-ckpt", default=None, help="optional path to old model checkpoint")
    p.add_argument("--new-ckpt", default=None, help="optional path to new model checkpoint")
    p.add_argument("--old-label", default=None, help="optional label for old model panel")
    p.add_argument("--new-label", default=None, help="optional label for new model panel")
    p.add_argument("--num-tiles", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--panel-width", type=int, default=720, help="width of each sub-panel after resize")
    p.add_argument("--output-dir", default=None, help="output directory")
    return p.parse_args()


def convert_from_color(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for rgb, cls in ISPRS_PALETTE_RGB2CLS.items():
        m = np.all(arr_3d == np.array(rgb).reshape(1, 1, 3), axis=2)
        arr_2d[m] = cls
    return arr_2d


def convert_to_color(arr_2d):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for cls, rgb in ISPRS_PALETTE_CLS2RGB.items():
        arr_3d[arr_2d == cls] = rgb
    return arr_3d


def get_protocol(dataset: str, data_root: str):
    root = os.path.join(data_root, dataset)
    if dataset == "Potsdam":
        return {
            "test_ids": ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"],
            "data_t": os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif"),
            "dsm_t": os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg"),
            "eroded_t": os.path.join(root, "5_Labels_for_participants_no_Boundary", "top_potsdam_{}_label_noBoundary.tif"),
        }
    raise ValueError(f"Unsupported target dataset: {dataset}")


def read_best(csv_path: str) -> Optional[BestMetric]:
    if not os.path.isfile(csv_path):
        return None
    best = None
    import csv

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get("target_mean_miou", "nan"))
                e = int(row.get("epoch", 0))
            except Exception:
                continue
            if math.isnan(m):
                continue
            if best is None or m > best.miou:
                best = BestMetric(epoch=e, miou=m)
    return best


def list_configs(run_dir: str) -> List[str]:
    configs = []
    for n in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, n)
        if os.path.isdir(p) and re.match(r"^c\d+_", n):
            configs.append(n)
    return configs


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for _ in sliding_window(top, step=step, window_size=window_size):
        c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def infer_tile(net, img, dsm, n_classes, window_size, stride, batch_size, device):
    pred = np.zeros(img.shape[:2] + (n_classes,), dtype=np.float32)
    total = int(math.ceil(count_sliding_window(img, step=stride, window_size=window_size) / float(batch_size)))

    with torch.no_grad():
        for coords in tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False):
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = torch.from_numpy(np.asarray(image_patches)).to(device)

            dmin, dmax = np.min(dsm), np.max(dsm)
            if dmax - dmin < 1e-8:
                dsm_norm = np.zeros_like(dsm, dtype=np.float32)
            else:
                dsm_norm = (dsm - dmin) / (dmax - dmin)
            dsm_patches = [np.copy(dsm_norm[x:x + w, y:y + h]) for x, y, w, h in coords]
            dsm_patches = torch.from_numpy(np.asarray(dsm_patches)).to(device)

            outs = net(image_patches, dsm_patches, mode="Test")
            outs = outs.detach().cpu().numpy()
            for out, (x, y, w, h) in zip(outs, coords):
                pred[x:x + w, y:y + h] += out.transpose((1, 2, 0))

    return np.argmax(pred, axis=-1)


def add_title(image: np.ndarray, title: str, bar_h: int = 34) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :, :] = 18
    canvas[bar_h:, :, :] = image
    pil_im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((max(4, (w - tw) // 2), max(2, (bar_h - th) // 2)), title, font=font, fill=(255, 255, 255))
    return np.asarray(pil_im)


def resize_panel(image: np.ndarray, target_w: int, is_label: bool) -> np.ndarray:
    pil = Image.fromarray(image)
    if pil.width == target_w:
        return np.asarray(pil)
    target_h = int(round(pil.height * (target_w / float(pil.width))))
    resample = Image.Resampling.NEAREST if is_label else Image.Resampling.LANCZOS
    pil = pil.resize((target_w, max(1, target_h)), resample=resample)
    return np.asarray(pil)


def hstack(panels: List[np.ndarray]) -> np.ndarray:
    max_h = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] < max_h:
            p = np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0)), mode="edge")
        out.append(p)
    return np.concatenate(out, axis=1)


def main():
    args = parse_args()

    old_run = os.path.abspath(args.old_run)
    new_run = os.path.abspath(args.new_run)
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(new_run, "weekly_report_assets_v2")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg_old = set(list_configs(old_run))
    cfg_new = set(list_configs(new_run))
    cfgs = sorted(cfg_old & cfg_new)
    if not cfgs:
        raise RuntimeError("No common config folders found between old-run and new-run.")

    rows = []
    for cfg in cfgs:
        old_csv = os.path.join(old_run, cfg, f"MFNet_week3_{args.source}2{args.target}_seed{args.seed}.csv")
        new_csv = os.path.join(new_run, cfg, f"MFNet_week3_{args.source}2{args.target}_seed{args.seed}.csv")
        old_best = read_best(old_csv)
        new_best = read_best(new_csv)
        if old_best is None or new_best is None:
            continue
        rows.append((cfg, old_best, new_best))

    if not rows:
        raise RuntimeError("No valid rows built from CSV files.")

    old_best_cfg = max(rows, key=lambda x: x[1].miou)[0]
    new_best_cfg = max(rows, key=lambda x: x[2].miou)[0]

    old_ckpt = args.old_ckpt or os.path.join(old_run, old_best_cfg, "UNetformer_week3_best.pth")
    new_ckpt = args.new_ckpt or os.path.join(new_run, new_best_cfg, "UNetformer_week3_best.pth")
    old_ckpt = os.path.abspath(old_ckpt)
    new_ckpt = os.path.abspath(new_ckpt)
    if not os.path.isfile(old_ckpt):
        raise FileNotFoundError(f"old checkpoint not found: {old_ckpt}")
    if not os.path.isfile(new_ckpt):
        raise FileNotFoundError(f"new checkpoint not found: {new_ckpt}")

    old_label = args.old_label or f"Old ({os.path.basename(os.path.dirname(old_ckpt))})"
    new_label = args.new_label or f"New ({os.path.basename(os.path.dirname(new_ckpt))})"

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mfnet_dir = os.path.join(repo_root, "MFNet")
    if mfnet_dir not in sys.path:
        sys.path.insert(0, mfnet_dir)

    sam_ckpt = os.path.join(mfnet_dir, "weights", "sam_vit_l_0b3195.pth")
    if os.path.isfile(sam_ckpt):
        os.environ["SSRS_MFNET_SAM_CKPT"] = sam_ckpt

    sys.argv = [sys.argv[0]]
    from UNetFormer_MMSAM import UNetFormer as MFNetModel  # pylint: disable=import-error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for preview generation.")

    proto = get_protocol(args.target, args.data_root)
    tile_ids = proto["test_ids"][: args.num_tiles]

    model_old = MFNetModel(num_classes=6).to(device)
    model_old.load_state_dict(torch.load(old_ckpt, map_location=device), strict=False)
    model_old.eval()

    model_new = MFNetModel(num_classes=6).to(device)
    model_new.load_state_dict(torch.load(new_ckpt, map_location=device), strict=False)
    model_new.eval()

    for tile_id in tile_ids:
        img = np.asarray(io.imread(proto["data_t"].format(tile_id)), dtype=np.float32)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        img_norm = rgb.astype(np.float32) / 255.0

        dsm = np.asarray(io.imread(proto["dsm_t"].format(tile_id)), dtype=np.float32)
        gt_rgb = np.asarray(io.imread(proto["eroded_t"].format(tile_id)), dtype=np.uint8)
        gt = convert_from_color(gt_rgb)

        pred_old = infer_tile(model_old, img_norm, dsm, 6, (256, 256), args.stride, args.batch_size, device)
        pred_new = infer_tile(model_new, img_norm, dsm, 6, (256, 256), args.stride, args.batch_size, device)

        rgb_p = resize_panel(rgb, args.panel_width, is_label=False)
        gt_p = resize_panel(convert_to_color(gt), args.panel_width, is_label=True)
        old_p = resize_panel(convert_to_color(pred_old), args.panel_width, is_label=True)
        new_p = resize_panel(convert_to_color(pred_new), args.panel_width, is_label=True)

        panel = hstack([
            add_title(rgb_p, "RGB"),
            add_title(gt_p, f"GT ({tile_id})"),
            add_title(old_p, old_label),
            add_title(new_p, new_label),
        ])

        out_path = os.path.join(out_dir, f"preview_tile_{tile_id}.png")
        Image.fromarray(panel).save(out_path, format="PNG", optimize=True, compress_level=9)
        print(f"[Saved] {out_path}")

    print(f"[Done] output_dir={out_dir}")


if __name__ == "__main__":
    main()
