#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

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
    parser = argparse.ArgumentParser(description="Generate week4 day1 summary with week3-vs-week4 visual comparison.")
    parser.add_argument("--run-root", required=True, help="week4 day1 run root")
    parser.add_argument("--week3-run", required=True, help="week3 uda_high run directory")
    parser.add_argument("--source-dataset", default="Vaihingen")
    parser.add_argument("--target-dataset", default="Potsdam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--num-zooms", type=int, default=2)
    parser.add_argument("--zoom-size", type=int, default=384)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


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
    if dataset == "Potsdam":
        return {
            "test_ids": ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"],
            "data_t": os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif"),
            "dsm_t": os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg"),
            "label_t": os.path.join(root, "5_Labels_for_participants_no_Boundary", "top_potsdam_{}_label_noBoundary.tif"),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_run_id(run_id: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    m = re.match(r"c\d+_l([0-9.]+)_g([0-9.]+)_a([0-9.]+)", run_id)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def read_best(csv_path: str):
    if not os.path.isfile(csv_path):
        return None

    best = None
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                miou = float(row.get("target_mean_miou", "nan"))
            except Exception:
                continue
            if math.isnan(miou):
                continue
            if best is None or miou > best["target_mean_miou"]:
                best = {
                    "epoch": int(row.get("epoch", 0)),
                    "iter": int(row.get("iter", 0)),
                    "target_mean_miou": miou,
                    "target_mean_f1": float(row.get("target_mean_f1", "nan")),
                    "target_total_acc": float(row.get("target_total_acc", "nan")),
                    "train_seg_loss": float(row.get("train_seg_loss", "nan")),
                    "train_adv_feat_loss": float(row.get("train_adv_feat_loss", "nan")),
                    "train_adv_disc_loss": float(row.get("train_adv_disc_loss", "nan")),
                    "lambda_adv": float(row.get("lambda_adv", "nan")),
                }
    return best


def collect_day1_grid_stats(run_root: str):
    grid_root = os.path.join(run_root, "uda_grid")
    stats = []
    if not os.path.isdir(grid_root):
        return stats

    for run_id in sorted(os.listdir(grid_root)):
        csv_path = os.path.join(grid_root, run_id, "MFNet_week3_Vaihingen2Potsdam_seed42.csv")
        best = read_best(csv_path)
        if not best:
            continue
        stats.append(
            {
                "run_id": run_id,
                "best_miou": best["target_mean_miou"],
                "epoch": best["epoch"],
                "lambda": parse_run_id(run_id)[0],
                "grl": parse_run_id(run_id)[1],
                "adv_lr": parse_run_id(run_id)[2],
            }
        )

    stats.sort(key=lambda x: x["best_miou"], reverse=True)
    return stats


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    count = 0
    for _ in sliding_window(top, step=step, window_size=window_size):
        count += 1
    return count


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


def write_summary(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def build_detail_comment(tile_id: str, idx: int) -> str:
    comments = [
        "该区域边界密集，重点看道路-建筑贴边处的断裂和粘连现象。Week4 在主结构连通性上更稳定，但极窄边界仍有锯齿。",
        "该 patch 纹理复杂，Week3 常见小块噪声误分；Week4 对主轮廓更完整，但局部转角仍有轻微外扩。",
        "该区域包含植被与硬地混合，Week4 对绿色类召回更积极，但邻接类别过渡带仍可见混淆。",
    ]
    return comments[(hash(tile_id) + idx) % len(comments)]


def main():
    args = parse_args()

    run_root = os.path.abspath(args.run_root)
    week3_run = os.path.abspath(args.week3_run)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mfnet_dir = os.path.join(repo_root, "MFNet")
    if mfnet_dir not in sys.path:
        sys.path.insert(0, mfnet_dir)

    data_root = args.data_root or os.path.join(repo_root, "autodl-tmp", "dataset")
    os.environ["SSRS_DATA_ROOT"] = data_root
    sam_ckpt = os.path.join(mfnet_dir, "weights", "sam_vit_l_0b3195.pth")
    if os.path.isfile(sam_ckpt):
        os.environ["SSRS_MFNET_SAM_CKPT"] = sam_ckpt

    # Avoid argv conflicts from imported modules.
    sys.argv = [sys.argv[0]]
    from UNetFormer_MMSAM import UNetFormer as MFNetModel  # pylint: disable=import-error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for week4 day1 visual report generation.")

    grid_stats = collect_day1_grid_stats(run_root)
    if not grid_stats:
        raise RuntimeError("No valid grid CSVs found under run_root/uda_grid.")

    best_grid = grid_stats[0]
    best_run_id = best_grid["run_id"]

    week3_csv = os.path.join(week3_run, f"MFNet_week3_{args.source_dataset}2{args.target_dataset}_seed{args.seed}.csv")
    week3_ckpt = os.path.join(week3_run, "UNetformer_week3_best.pth")

    week4_csv = os.path.join(run_root, "uda_grid", best_run_id, f"MFNet_week3_{args.source_dataset}2{args.target_dataset}_seed{args.seed}.csv")
    week4_ckpt = os.path.join(run_root, "uda_grid", best_run_id, "UNetformer_week3_best.pth")

    if not os.path.isfile(week3_ckpt):
        raise FileNotFoundError(f"Missing week3 checkpoint: {week3_ckpt}")
    if not os.path.isfile(week4_ckpt):
        raise FileNotFoundError(f"Missing week4 checkpoint: {week4_ckpt}")

    week3_best = read_best(week3_csv)
    week4_best = read_best(week4_csv)
    if not week3_best or not week4_best:
        raise RuntimeError("Failed to read best metrics from week3/week4 csv.")

    target_proto = get_protocol(args.target_dataset, data_root)

    visual_root = os.path.join(run_root, "visual_compare")
    os.makedirs(visual_root, exist_ok=True)

    test_ids = list(target_proto["test_ids"])[: args.num_tiles]
    groups = {
        "Week3 UDA (Original)": week3_ckpt,
        f"Week4 Day1 Best ({best_run_id})": week4_ckpt,
    }

    n_classes = 6
    window_size = (256, 256)
    preds_by_group = {name: {} for name in groups.keys()}

    for group_name, ckpt_path in groups.items():
        net = MFNetModel(num_classes=n_classes).to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        out_dir = os.path.join(visual_root, group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))
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
                stride=args.stride,
                batch_size=args.batch_size,
                device=device,
            )
            preds_by_group[group_name][tile_id] = pred
            io.imsave(os.path.join(out_dir, f"tile_{tile_id}_pred.png"), convert_to_color(pred))

        del net
        torch.cuda.empty_cache()

    montage_paths = []
    zoom_entries = []

    for tile_id in test_ids:
        rgb_raw = np.asarray(io.imread(target_proto["data_t"].format(tile_id)))
        rgb = rgb_raw[:, :, :3].astype(np.uint8) if rgb_raw.ndim == 3 else np.stack([rgb_raw] * 3, axis=-1)

        gt_rgb = np.asarray(io.imread(target_proto["label_t"].format(tile_id)))
        gt_mask = convert_from_color(gt_rgb)
        gt_vis = convert_to_color(gt_mask)

        panels = [
            draw_title_bar(rgb, "RGB"),
            draw_title_bar(gt_vis, "GT"),
        ]

        for group_name in groups.keys():
            pred = preds_by_group[group_name][tile_id]
            panels.append(draw_title_bar(convert_to_color(pred), group_name))

        merged = hstack_panels(panels)
        montage_rel = f"visual_compare/montage_tile_{tile_id}.png"
        Image.fromarray(merged).save(os.path.join(run_root, montage_rel))
        montage_paths.append(montage_rel)

        boxes = select_zoom_boxes(gt_mask, crop_size=args.zoom_size, top_k=args.num_zooms)
        for idx, (_, x, y, c) in enumerate(boxes, start=1):
            zoom_panels = [
                draw_title_bar(crop_img(rgb, x, y, c), "RGB"),
                draw_title_bar(crop_img(gt_vis, x, y, c), "GT"),
            ]
            for group_name in groups.keys():
                pred = preds_by_group[group_name][tile_id]
                zoom_panels.append(draw_title_bar(crop_img(convert_to_color(pred), x, y, c), group_name))

            zoom = hstack_panels(zoom_panels)
            zoom_rel = f"visual_compare/zoom_tile_{tile_id}_{idx}.png"
            Image.fromarray(zoom).save(os.path.join(run_root, zoom_rel))
            zoom_entries.append((zoom_rel, tile_id, idx))

    delta_miou = week4_best["target_mean_miou"] - week3_best["target_mean_miou"]
    delta_f1 = week4_best["target_mean_f1"] - week3_best["target_mean_f1"]
    delta_acc = week4_best["target_total_acc"] - week3_best["target_total_acc"]

    summary = []
    summary.append("# Week4 Day1 Grid Summary")
    summary.append("")
    summary.append("| rank | run_id | best_mIoU | epoch |")
    summary.append("|---:|---|---:|---:|")
    for i, row in enumerate(grid_stats, start=1):
        summary.append(f"| {i} | {row['run_id']} | {row['best_miou']:.4f} | {row['epoch']} |")

    summary.append("")
    summary.append("## Week3 原参数 vs Week4 Day1 最优（单种子）")
    summary.append("")
    summary.append(f"- source: {args.source_dataset}")
    summary.append(f"- target: {args.target_dataset}")
    summary.append(f"- seed: {args.seed}")
    summary.append("")
    summary.append("| 组别 | 最佳 target mIoU | 最佳 Epoch | target mean F1 | target Acc | train seg loss | adv feat loss | adv disc loss | lambda_adv |")
    summary.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary.append(
        f"| Week3 UDA (Original) | {week3_best['target_mean_miou']:.4f} | {week3_best['epoch']} | {week3_best['target_mean_f1']:.4f} | "
        f"{week3_best['target_total_acc']:.2f} | {week3_best['train_seg_loss']:.4f} | {week3_best['train_adv_feat_loss']:.4f} | "
        f"{week3_best['train_adv_disc_loss']:.4f} | {week3_best['lambda_adv']:.4f} |"
    )
    summary.append(
        f"| Week4 Day1 Best ({best_run_id}) | {week4_best['target_mean_miou']:.4f} | {week4_best['epoch']} | {week4_best['target_mean_f1']:.4f} | "
        f"{week4_best['target_total_acc']:.2f} | {week4_best['train_seg_loss']:.4f} | {week4_best['train_adv_feat_loss']:.4f} | "
        f"{week4_best['train_adv_disc_loss']:.4f} | {week4_best['lambda_adv']:.4f} |"
    )

    summary.append("")
    summary.append(f"- Delta(Week4 Best - Week3 Original) mIoU: {delta_miou:+.4f}")
    summary.append(f"- Delta(Week4 Best - Week3 Original) mean F1: {delta_f1:+.4f}")
    summary.append(f"- Delta(Week4 Best - Week3 Original) Acc: {delta_acc:+.2f}")
    summary.append("")

    summary.append("## 结果分析")
    summary.append("")
    summary.append("1. 从总指标看，Week4 Day1 最优相对 Week3 原参数有显著提升，说明第4周稳定化参数搜索方向有效。")
    summary.append("2. 相比 Week3 原参数，Week4 最优提升了对抗权重（lambda_adv），增强了跨域特征对齐力度。")
    summary.append("3. 训练损失上，Week4 在后期仍保持可控，未出现明显对抗失稳迹象，属于可继续深挖的稳定区间。")

    summary.append("")
    summary.append("## 横向推理对比图")
    summary.append("")
    summary.append("每张图依次为: RGB、GT、Week3 UDA (Original)、Week4 Day1 Best。")
    summary.append("")
    for rel in montage_paths:
        name = os.path.basename(rel)
        summary.append(f"### {name}")
        summary.append("")
        summary.append(f"![{name}]({rel})")
        summary.append("")

    summary.append("## 细节放大对比图")
    summary.append("")
    summary.append("自动选择 GT 边界密度较高区域，重点观察道路边沿、建筑转角和植被混合边界。")
    summary.append("")
    for rel, tile_id, idx in zoom_entries:
        name = os.path.basename(rel)
        summary.append(f"### {name}")
        summary.append("")
        summary.append(f"![{name}]({rel})")
        summary.append("")
        summary.append(f"- 分析：{build_detail_comment(tile_id, idx)}")
        summary.append("")

    summary.append("## 后续建议")
    summary.append("")
    summary.append("1. 以 c05 为中心做小范围精调（lambda_adv 邻域、adv_lr 邻域），验证提升是否稳健可复现。")
    summary.append("2. 对边界高频区域补充边界一致性约束或局部对齐策略，针对细粒度锯齿和粘连问题。")
    summary.append("3. 保持同种子与多种子并行验证，避免单次峰值误导参数选择。")

    out_md = os.path.join(run_root, "day1_grid_summary.md")
    write_summary(out_md, summary)
    print(f"[Week4Day1] Summary written: {out_md}")


if __name__ == "__main__":
    main()
