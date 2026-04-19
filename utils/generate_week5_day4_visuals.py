#!/usr/bin/env python3
import argparse
import itertools
import math
import os
import sys
from typing import Dict, List

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


def parse_args():
    p = argparse.ArgumentParser(description="Generate Week5 Day4 visuals (F3-v2 and F5-v1).")
    p.add_argument("--run-root", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--target-dataset", default="Potsdam")
    p.add_argument("--num-tiles", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--num-zooms", type=int, default=3)
    p.add_argument("--zoom-size", type=int, default=384)
    p.add_argument("--max-panel-width", type=int, default=3200)
    p.add_argument("--png-compress-level", type=int, default=9)
    return p.parse_args()


def get_protocol(dataset: str, data_root: str):
    root = os.path.join(data_root, dataset)
    if dataset == "Potsdam":
        return {
            "test_ids": ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"],
            "data_t": os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif"),
            "dsm_t": os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg"),
            "label_t": os.path.join(root, "5_Labels_for_participants_no_Boundary", "top_potsdam_{}_label_noBoundary.tif"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


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


def save_png(path: str, image: np.ndarray, compress_level: int, max_width: int = 0):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image)
    if max_width > 0 and pil.width > max_width:
        new_h = int(round(pil.height * (max_width / float(pil.width))))
        pil = pil.resize((max_width, max(1, new_h)), resample=Image.Resampling.BILINEAR)
    level = max(0, min(9, int(compress_level)))
    pil.save(path, format="PNG", optimize=True, compress_level=level)


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


def infer_logits_avg(net, img, dsm, n_classes, window_size, stride, batch_size, device):
    logits_sum = np.zeros(img.shape[:2] + (n_classes,), dtype=np.float32)
    count_map = np.zeros(img.shape[:2] + (1,), dtype=np.float32)
    total = int(math.ceil(count_sliding_window(img, step=stride, window_size=window_size) / float(batch_size)))

    dmin, dmax = np.min(dsm), np.max(dsm)
    if dmax - dmin < 1e-8:
        dsm_norm = np.zeros_like(dsm, dtype=np.float32)
    else:
        dsm_norm = (dsm - dmin) / (dmax - dmin)

    with torch.no_grad():
        for coords in tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False):
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            dsm_patches = [np.copy(dsm_norm[x:x + w, y:y + h]) for x, y, w, h in coords]
            image_t = torch.from_numpy(np.asarray(image_patches)).to(device)
            dsm_t = torch.from_numpy(np.asarray(dsm_patches)).to(device)

            outs = net(image_t, dsm_t, mode="Test").detach().cpu().numpy()
            for out, (x, y, w, h) in zip(outs, coords):
                logits_sum[x:x + w, y:y + h] += out.transpose((1, 2, 0))
                count_map[x:x + w, y:y + h] += 1.0

    count_map = np.maximum(count_map, 1e-6)
    return logits_sum / count_map


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-8)


def add_title(image: np.ndarray, title: str, bar_h: int = 36) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, _ = image.shape
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


def hstack_panels(panels: List[np.ndarray]) -> np.ndarray:
    max_h = max([p.shape[0] for p in panels])
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            p = np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0)), mode="edge")
        padded.append(p)
    return np.concatenate(padded, axis=1)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    b = np.zeros((h, w), dtype=np.uint8)
    b[1:, :] |= (mask[1:, :] != mask[:-1, :])
    b[:, 1:] |= (mask[:, 1:] != mask[:, :-1])
    return b


def select_zoom_boxes(gt_mask: np.ndarray, crop_size: int, top_k: int) -> List[tuple]:
    h, w = gt_mask.shape
    crop = max(64, min(crop_size, h, w))
    step = max(32, crop // 3)

    b = mask_boundary(gt_mask)
    candidates = []
    for y in range(0, h - crop + 1, step):
        for x in range(0, w - crop + 1, step):
            candidates.append((int(np.sum(b[y:y + crop, x:x + crop])), x, y, crop))

    if not candidates:
        return [(0, 0, 0, crop)]

    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = []
    min_dist = crop // 2
    for score, x, y, c in candidates:
        keep = True
        cx, cy = x + c // 2, y + c // 2
        for _, sx, sy, sc in selected:
            if abs(cx - (sx + sc // 2)) + abs(cy - (sy + sc // 2)) < min_dist:
                keep = False
                break
        if keep:
            selected.append((score, x, y, c))
        if len(selected) >= top_k:
            break

    return selected if selected else [candidates[0]]


def crop_img(img: np.ndarray, x: int, y: int, c: int) -> np.ndarray:
    return np.ascontiguousarray(img[y:y + c, x:x + c])


def scalar_to_heatmap01(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    r = (v * 255.0).astype(np.uint8)
    g = ((1.0 - np.abs(v - 0.5) * 2.0) * 255.0).astype(np.uint8)
    b = ((1.0 - v) * 255.0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main():
    args = parse_args()
    run_root = os.path.abspath(args.run_root)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mfnet_dir = os.path.join(repo_root, "MFNet")
    if mfnet_dir not in sys.path:
        sys.path.insert(0, mfnet_dir)

    data_root = args.data_root or os.path.join(repo_root, "autodl-tmp", "dataset")
    os.environ["SSRS_DATA_ROOT"] = data_root
    sam_ckpt = os.path.join(mfnet_dir, "weights", "sam_vit_l_0b3195.pth")
    if os.path.isfile(sam_ckpt):
        os.environ["SSRS_MFNET_SAM_CKPT"] = sam_ckpt

    sys.argv = [sys.argv[0]]
    from UNetFormer_MMSAM import UNetFormer as MFNetModel  # pylint: disable=import-error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for week5 day4 visual generation.")

    proto = get_protocol(args.target_dataset, data_root)
    tile_ids = list(proto["test_ids"])[: args.num_tiles]

    groups_f3: Dict[str, str] = {
        "Source-only": "source_only_seed42",
        "+UDA": "uda_seed42",
        "+UDA+OBJ": "uda_obj_seed42",
        "+UDA+BDY+OBJ": "uda_bdy_obj_seed42",
    }

    models: Dict[str, torch.nn.Module] = {}
    for label, run_id in groups_f3.items():
        ckpt = os.path.join(run_root, run_id, "UNetformer_week3_best.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"missing checkpoint for {label}: {ckpt}")
        net = MFNetModel(num_classes=6).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        net.eval()
        models[label] = net

    f3_dir = os.path.join(run_root, "visual_f3_v2")
    f5_dir = os.path.join(run_root, "pseudo_f5_v1")
    os.makedirs(f3_dir, exist_ok=True)
    os.makedirs(f5_dir, exist_ok=True)

    pred_cache: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in models.keys()}
    conf_cache: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in models.keys()}

    conf_summary = []
    for tile_id in tile_ids:
        img = np.asarray(io.imread(proto["data_t"].format(tile_id)), dtype=np.float32)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        img = rgb.astype(np.float32) / 255.0

        dsm = np.asarray(io.imread(proto["dsm_t"].format(tile_id)), dtype=np.float32)
        gt_rgb = np.asarray(io.imread(proto["label_t"].format(tile_id)), dtype=np.uint8)
        gt = convert_from_color(gt_rgb)

        for label, net in models.items():
            logits = infer_logits_avg(net, img, dsm, 6, (256, 256), args.stride, args.batch_size, device)
            probs = softmax_np(logits, axis=-1)
            pred_cache[label][tile_id] = np.argmax(probs, axis=-1).astype(np.uint8)
            conf_cache[label][tile_id] = np.max(probs, axis=-1).astype(np.float32)

        f3_panels = [
            add_title(rgb, "RGB"),
            add_title(convert_to_color(gt), f"GT ({tile_id})"),
        ]
        for label in groups_f3.keys():
            f3_panels.append(add_title(convert_to_color(pred_cache[label][tile_id]), label))

        f3_montage = hstack_panels(f3_panels)
        save_png(os.path.join(f3_dir, f"montage_tile_{tile_id}.png"), f3_montage, args.png_compress_level, args.max_panel_width)

        boxes = select_zoom_boxes(gt, crop_size=args.zoom_size, top_k=args.num_zooms)
        for idx, (_, x, y, c) in enumerate(boxes, start=1):
            z_panels = [
                add_title(crop_img(rgb, x, y, c), "RGB"),
                add_title(crop_img(convert_to_color(gt), x, y, c), "GT"),
            ]
            for label in groups_f3.keys():
                z_panels.append(add_title(crop_img(convert_to_color(pred_cache[label][tile_id]), x, y, c), label))
            save_png(os.path.join(f3_dir, f"zoom_tile_{tile_id}_{idx}.png"), hstack_panels(z_panels), args.png_compress_level, args.max_panel_width)

        pred_uda = pred_cache["+UDA"][tile_id]
        pred_obj = pred_cache["+UDA+OBJ"][tile_id]
        conf_obj = conf_cache["+UDA+OBJ"][tile_id]

        agree = (pred_uda == pred_obj).astype(np.float32)
        pseudo_quality = np.clip(conf_obj * (0.5 + 0.5 * agree), 0.0, 1.0)
        boundary = mask_boundary(pred_obj).astype(np.float32)
        boundary_response = np.clip(0.6 * boundary + 0.4 * (1.0 - conf_obj), 0.0, 1.0)

        conf_summary.append((tile_id, float(np.mean(conf_obj)), float(np.mean(pseudo_quality)), float(np.mean(boundary_response))))

        f5_panels = [
            add_title(rgb, "RGB"),
            add_title(convert_to_color(pred_uda), "+UDA pred"),
            add_title(convert_to_color(pred_obj), "+UDA+OBJ pred"),
            add_title(scalar_to_heatmap01(conf_obj), "Conf(+UDA+OBJ)"),
            add_title(scalar_to_heatmap01(pseudo_quality), "Pseudo Quality"),
            add_title(scalar_to_heatmap01(boundary_response), "Boundary Response"),
        ]
        save_png(os.path.join(f5_dir, f"pseudo_tile_{tile_id}_pUDApOBJ.png"), hstack_panels(f5_panels), args.png_compress_level, args.max_panel_width)

    for _, m in models.items():
        del m
    torch.cuda.empty_cache()

    summary_md = os.path.join(run_root, "week5_day4_visual_summary.md")
    lines: List[str] = []
    lines.append("# Week5 Day4 Visual Summary")
    lines.append("")
    lines.append("## 交付清单")
    lines.append("")
    lines.append(f"- F3-v2 目录: visual_f3_v2（{len(tile_ids)} 张全图 + {len(tile_ids) * args.num_zooms} 张局部放大）")
    lines.append(f"- F5-v1 目录: pseudo_f5_v1（{len(tile_ids)} 张伪标签质量/边界响应图）")
    lines.append("")
    lines.append("## F3-v2（边界放大）")
    lines.append("")
    for tile_id in tile_ids:
        lines.append(f"### montage_tile_{tile_id}.png")
        lines.append("")
        lines.append(f"![montage_tile_{tile_id}](visual_f3_v2/montage_tile_{tile_id}.png)")
        lines.append("")

    lines.append("## F5-v1（伪标签质量/边界响应）")
    lines.append("")
    lines.append("| tile | mean_conf | mean_pseudo_quality | mean_boundary_response |")
    lines.append("|---|---:|---:|---:|")
    for tile_id, c1, c2, c3 in conf_summary:
        lines.append(f"| {tile_id} | {c1:.4f} | {c2:.4f} | {c3:.4f} |")

    lines.append("")
    for tile_id in tile_ids:
        lines.append(f"### pseudo_tile_{tile_id}_pUDApOBJ.png")
        lines.append("")
        lines.append(f"![pseudo_tile_{tile_id}](pseudo_f5_v1/pseudo_tile_{tile_id}_pUDApOBJ.png)")
        lines.append("")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[Week5Day4] summary written: {summary_md}")
    print(f"[Week5Day4] F3 dir: {f3_dir}")
    print(f"[Week5Day4] F5 dir: {f5_dir}")


if __name__ == "__main__":
    main()
