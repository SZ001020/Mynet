#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import os
import sys
from typing import Dict, List, Tuple

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
    p = argparse.ArgumentParser(description="Generate week3 weak-cross summary with visual comparisons.")
    p.add_argument("--run-root", required=True, help="week3 run root, e.g. runs/week3_weak_cross/<timestamp>")
    p.add_argument("--data-root", default=None, help="dataset root, default: <repo>/autodl-tmp/dataset")
    p.add_argument("--source-dataset", default="Vaihingen", help="source dataset name")
    p.add_argument("--target-dataset", default="Potsdam", help="target dataset name")
    p.add_argument("--seed", type=int, default=42, help="seed for CSV filename lookup")
    p.add_argument("--num-tiles", type=int, default=3, help="how many target test tiles to visualize")
    p.add_argument("--batch-size", type=int, default=8, help="inference patch batch size")
    p.add_argument("--stride", type=int, default=32, help="inference sliding stride")
    p.add_argument("--num-zooms", type=int, default=2, help="zoom crops per tile")
    p.add_argument("--zoom-size", type=int, default=384, help="zoom crop side length")
    p.add_argument("--max-panel-width", type=int, default=3200, help="max width for montage/zoom PNGs")
    p.add_argument("--png-compress-level", type=int, default=9, help="PNG compress level (0-9)")
    return p.parse_args()


def convert_from_color(arr_3d, palette=ISPRS_PALETTE_RGB2CLS):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for rgb, cls in palette.items():
        m = np.all(arr_3d == np.array(rgb).reshape(1, 1, 3), axis=2)
        arr_2d[m] = cls
    return arr_2d


def convert_to_color(arr_2d, palette=ISPRS_PALETTE_CLS2RGB):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for cls, rgb in palette.items():
        arr_3d[arr_2d == cls] = rgb
    return arr_3d


def get_protocol(dataset: str, data_root: str):
    root = os.path.join(data_root, dataset)
    if dataset == "Vaihingen":
        return {
            "test_ids": ["5", "21", "15", "30"],
            "data_t": os.path.join(root, "top", "top_mosaic_09cm_area{}.tif"),
            "dsm_t": os.path.join(root, "dsm", "dsm_09cm_matching_area{}.tif"),
            "eroded_t": os.path.join(root, "gts_eroded_for_participants", "top_mosaic_09cm_area{}_noBoundary.tif"),
            "stride": 32,
        }
    if dataset == "Potsdam":
        return {
            "test_ids": ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"],
            "data_t": os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif"),
            "dsm_t": os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg"),
            "eroded_t": os.path.join(root, "5_Labels_for_participants_no_Boundary", "top_potsdam_{}_label_noBoundary.tif"),
            "stride": 128,
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


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


def draw_title_bar(image: np.ndarray, title: str, bar_h: int = 40) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    h, w, _ = image.shape
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :, :] = 20
    canvas[bar_h:, :, :] = image

    pil_im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(4, (w - tw) // 2)
    ty = max(2, (bar_h - th) // 2)
    draw.text((tx, ty), title, font=font, fill=(255, 255, 255))
    return np.asarray(pil_im)


def hstack_panels(panels: List[np.ndarray]) -> np.ndarray:
    max_h = max([p.shape[0] for p in panels])
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            p = np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0)), mode="constant", constant_values=0)
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
            score = int(np.sum(b[y:y + crop, x:x + crop]))
            candidates.append((score, x, y, crop))
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


def save_png(path: str, image: np.ndarray, compress_level: int, max_width: int = 0):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image)
    if max_width > 0 and pil.width > max_width:
        new_h = int(round(pil.height * (max_width / float(pil.width))))
        pil = pil.resize((max_width, max(1, new_h)), resample=Image.Resampling.BILINEAR)
    level = max(0, min(9, int(compress_level)))
    pil.save(path, format="PNG", optimize=True, compress_level=level)


def read_best(csv_path: str):
    if not os.path.isfile(csv_path):
        return None
    best = None
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                metric = float(row.get("target_mean_miou", "nan"))
            except Exception:
                continue
            if math.isnan(metric):
                continue
            if best is None or metric > best["target_mean_miou"]:
                best = {
                    "epoch": row.get("epoch", "NA"),
                    "iter": row.get("iter", "NA"),
                    "target_mean_miou": metric,
                    "target_mean_f1": float(row.get("target_mean_f1", "nan")),
                    "target_total_acc": float(row.get("target_total_acc", "nan")),
                    "train_seg_loss": float(row.get("train_seg_loss", "nan")),
                    "train_adv_feat_loss": float(row.get("train_adv_feat_loss", "nan")),
                    "train_adv_disc_loss": float(row.get("train_adv_disc_loss", "nan")),
                }
    return best


def write_report(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


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

    # avoid parse_args side-effects inside imported modules
    sys.argv = [sys.argv[0]]
    from UNetFormer_MMSAM import UNetFormer as MFNetModel  # pylint: disable=import-error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for week3 visual report generation.")

    source_name = args.source_dataset
    target_name = args.target_dataset
    target_proto = get_protocol(target_name, data_root)

    groups = {
        "Source-only": "source_only",
        "Single-level UDA": "uda_high",
    }

    metrics = {}
    checkpoints = {}
    for group_name, subdir in groups.items():
        gdir = os.path.join(run_root, subdir)
        csv_name = f"MFNet_week3_{source_name}2{target_name}_seed{args.seed}.csv"
        csv_path = os.path.join(gdir, csv_name)
        ckpt_path = os.path.join(gdir, "UNetformer_week3_best.pth")
        metrics[group_name] = read_best(csv_path)
        checkpoints[group_name] = ckpt_path if os.path.isfile(ckpt_path) else None

    visual_root = os.path.join(run_root, "visual_compare")
    os.makedirs(visual_root, exist_ok=True)

    test_ids = list(target_proto["test_ids"])[: args.num_tiles]
    n_classes = 6
    window_size = (256, 256)
    stride = args.stride if args.stride > 0 else int(target_proto["stride"])

    preds_by_group = {k: {} for k in groups.keys()}

    for group_name in groups.keys():
        ckpt = checkpoints[group_name]
        if ckpt is None:
            print(f"[Week3Report][WARN] missing checkpoint for {group_name}")
            continue

        net = MFNetModel(num_classes=n_classes).to(device)
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        out_dir = os.path.join(visual_root, groups[group_name])
        os.makedirs(out_dir, exist_ok=True)

        for tile_id in test_ids:
            img = np.asarray(io.imread(target_proto["data_t"].format(tile_id)), dtype="float32")
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            img = img / 255.0

            dsm = np.asarray(io.imread(target_proto["dsm_t"].format(tile_id)), dtype="float32")
            pred = infer_tile(
                net,
                img,
                dsm,
                n_classes=n_classes,
                window_size=window_size,
                stride=stride,
                batch_size=args.batch_size,
                device=device,
            )
            preds_by_group[group_name][tile_id] = pred
            save_png(os.path.join(out_dir, f"tile_{tile_id}_pred.png"), convert_to_color(pred), compress_level=args.png_compress_level)

        del net
        torch.cuda.empty_cache()

    montage_paths = []
    zoom_paths = []

    for tile_id in test_ids:
        rgb_raw = np.asarray(io.imread(target_proto["data_t"].format(tile_id)))
        rgb = rgb_raw[:, :, :3].astype(np.uint8) if rgb_raw.ndim == 3 else np.stack([rgb_raw] * 3, axis=-1)

        gt_rgb = np.asarray(io.imread(target_proto["eroded_t"].format(tile_id)))
        gt_mask = convert_from_color(gt_rgb)
        gt_vis = convert_to_color(gt_mask)

        panel_imgs = [
            draw_title_bar(rgb, "RGB"),
            draw_title_bar(gt_vis, "GT"),
        ]

        for group_name in groups.keys():
            pred = preds_by_group.get(group_name, {}).get(tile_id)
            if pred is None:
                pred_vis = np.zeros_like(gt_vis, dtype=np.uint8)
            else:
                pred_vis = convert_to_color(pred)
            panel_imgs.append(draw_title_bar(pred_vis, group_name))

        merged = hstack_panels(panel_imgs)
        rel = f"visual_compare/montage_tile_{tile_id}.png"
        save_png(os.path.join(run_root, rel), merged, compress_level=args.png_compress_level, max_width=args.max_panel_width)
        montage_paths.append(rel)

        boxes = select_zoom_boxes(gt_mask, crop_size=args.zoom_size, top_k=args.num_zooms)
        for idx, (_, x, y, c) in enumerate(boxes, start=1):
            zoom_imgs = [
                draw_title_bar(crop_img(rgb, x, y, c), "RGB"),
                draw_title_bar(crop_img(gt_vis, x, y, c), "GT"),
            ]
            for group_name in groups.keys():
                pred = preds_by_group.get(group_name, {}).get(tile_id)
                if pred is None:
                    pred_vis = np.zeros_like(gt_vis, dtype=np.uint8)
                else:
                    pred_vis = convert_to_color(pred)
                zoom_imgs.append(draw_title_bar(crop_img(pred_vis, x, y, c), group_name))

            zoom = hstack_panels(zoom_imgs)
            rel = f"visual_compare/zoom_tile_{tile_id}_{idx}.png"
            save_png(os.path.join(run_root, rel), zoom, compress_level=args.png_compress_level, max_width=args.max_panel_width)
            zoom_paths.append(rel)

    report = []
    report.append("# 第3周 weak-cross-domain 原型结果总结（单种子）")
    report.append("")
    report.append(f"- source: {source_name}")
    report.append(f"- target: {target_name}")
    report.append(f"- seed: {args.seed}")
    report.append("")
    report.append("| 组别 | 最佳 target mIoU | 最佳 Epoch | target mean F1 | target Acc | train seg loss | adv feat loss | adv disc loss |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    src = metrics.get("Source-only")
    uda = metrics.get("Single-level UDA")

    def row(name, st):
        if not st:
            return f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        return (
            f"| {name} | {st['target_mean_miou']:.4f} | {st['epoch']} | {st['target_mean_f1']:.4f} | "
            f"{st['target_total_acc']:.2f} | {st['train_seg_loss']:.4f} | {st['train_adv_feat_loss']:.4f} | {st['train_adv_disc_loss']:.4f} |"
        )

    report.append(row("Source-only", src))
    report.append(row("Single-level UDA", uda))

    report.append("")
    if src and uda:
        delta = uda["target_mean_miou"] - src["target_mean_miou"]
        report.append(f"- Delta(UDA - Source-only): {delta:+.4f}")
        if delta > 0:
            report.append("- 结论: 第3周原型门槛达成（UDA 相对 source-only 为正向）。")
        else:
            report.append("- 结论: 第3周原型门槛未达成（UDA 未优于 source-only），建议进入第4周前先调 lambda_adv / grl / adv_lr。")
    report.append("")

    report.append("## 横向推理对比图")
    report.append("")
    report.append("每张图依次为: RGB、GT、Source-only、Single-level UDA。")
    report.append("")
    for p in montage_paths:
        report.append(f"### {os.path.basename(p)}")
        report.append("")
        report.append(f"![{os.path.basename(p)}]({p})")
        report.append("")

    report.append("## 细节放大对比图")
    report.append("")
    report.append("自动选择 GT 边界密度较高区域，重点观察道路边沿与建筑转角。")
    report.append("")
    for p in zoom_paths:
        report.append(f"### {os.path.basename(p)}")
        report.append("")
        report.append(f"![{os.path.basename(p)}]({p})")
        report.append("")

    report.append("## 分析建议")
    report.append("")
    report.append("1. 若 UDA 在局部边界更平滑但总分未涨，优先在第4周调稳定性参数而非直接否定对抗分支。")
    report.append("2. 若 UDA 在总体和局部都退化，先降低 lambda_adv 或 grl_lambda，再观察判别器损失是否过强。")
    report.append("3. 第3周阶段先不叠加结构项与 prompt 分支，保持变量可解释。")

    report_path = os.path.join(run_root, "week3_weak_cross_summary.md")
    write_report(report_path, report)
    print(f"[Week3Report] Summary written: {report_path}")


if __name__ == "__main__":
    main()
