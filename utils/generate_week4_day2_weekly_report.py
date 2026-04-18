#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
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


@dataclass
class BestMetric:
    epoch: int
    miou: float
    f1: float
    acc: float
    seg_loss: float
    adv_feat_loss: float
    adv_disc_loss: float
    lambda_adv: float


def parse_args():
    p = argparse.ArgumentParser(description="Generate week4 day2 weekly report from two day2 refine runs.")
    p.add_argument("--old-run", required=True, help="old day2 run dir, e.g. .../20260412_002225")
    p.add_argument("--new-run", required=True, help="new day2 run dir, e.g. .../20260413_165257")
    p.add_argument("--data-root", default="/root/Mynet/autodl-tmp/dataset", help="dataset root")
    p.add_argument("--source", default="Vaihingen")
    p.add_argument("--target", default="Potsdam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-tiles", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-panel-width", type=int, default=3200, help="max output width for comparison panel")
    p.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality (1-95), lower means smaller files")
    p.add_argument("--output-md", default=None, help="output markdown path")
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
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get("target_mean_miou", "nan"))
            except Exception:
                continue
            if math.isnan(m):
                continue
            if best is None or m > best.miou:
                best = BestMetric(
                    epoch=int(row.get("epoch", 0)),
                    miou=m,
                    f1=float(row.get("target_mean_f1", "nan")),
                    acc=float(row.get("target_total_acc", "nan")),
                    seg_loss=float(row.get("train_seg_loss", "nan")),
                    adv_feat_loss=float(row.get("train_adv_feat_loss", "nan")),
                    adv_disc_loss=float(row.get("train_adv_disc_loss", "nan")),
                    lambda_adv=float(row.get("lambda_adv", "nan")),
                )
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


def draw_title_bar(image: np.ndarray, title: str, bar_h: int = 36) -> np.ndarray:
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


def compress_panel(panel: np.ndarray, max_width: int) -> Image.Image:
    img = Image.fromarray(panel)
    if img.width > max_width:
        new_h = int(round(img.height * (max_width / float(img.width))))
        img = img.resize((max_width, max(1, new_h)), resample=Image.Resampling.BILINEAR)
    return img


def main():
    args = parse_args()
    old_run = os.path.abspath(args.old_run)
    new_run = os.path.abspath(args.new_run)

    if args.output_md:
        output_md = os.path.abspath(args.output_md)
    else:
        output_md = os.path.join(new_run, "week4_day2_weekly_report.md")

    assets_dir = os.path.join(os.path.dirname(output_md), "weekly_report_assets")
    os.makedirs(assets_dir, exist_ok=True)

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
        rows.append((cfg, old_best, new_best, new_best.miou - old_best.miou))

    if not rows:
        raise RuntimeError("No valid CSV comparison rows were built.")

    rows.sort(key=lambda x: x[3], reverse=True)

    old_best_cfg = max(rows, key=lambda x: x[1].miou)[0]
    new_best_cfg = max(rows, key=lambda x: x[2].miou)[0]

    old_ckpt = os.path.join(old_run, old_best_cfg, "UNetformer_week3_best.pth")
    new_ckpt = os.path.join(new_run, new_best_cfg, "UNetformer_week3_best.pth")

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
        raise RuntimeError("CUDA is required for inference comparison generation.")

    proto = get_protocol(args.target, args.data_root)
    tile_ids = proto["test_ids"][: args.num_tiles]

    model_old = MFNetModel(num_classes=6).to(device)
    model_old.load_state_dict(torch.load(old_ckpt, map_location=device), strict=False)
    model_old.eval()

    model_new = MFNetModel(num_classes=6).to(device)
    model_new.load_state_dict(torch.load(new_ckpt, map_location=device), strict=False)
    model_new.eval()

    visual_list = []
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

        old_c = convert_to_color(pred_old)
        new_c = convert_to_color(pred_new)

        panel = hstack_panels([
            draw_title_bar(rgb, "RGB"),
            draw_title_bar(convert_to_color(gt), f"GT ({tile_id})"),
            draw_title_bar(old_c, f"Round1 Best ({old_best_cfg})"),
            draw_title_bar(new_c, f"Round2 Best ({new_best_cfg})"),
        ])

        out_name = f"compare_tile_{tile_id}.jpg"
        out_path = os.path.join(assets_dir, out_name)
        panel_img = compress_panel(panel, args.max_panel_width)
        panel_img.save(out_path, format="JPEG", quality=max(1, min(95, args.jpeg_quality)), optimize=True, progressive=True)
        visual_list.append(out_name)

    # markdown
    lines: List[str] = []
    lines.append("# Week4 Day2 Refine 周报")
    lines.append("")
    lines.append("## 一、总体结论")
    lines.append("")
    lines.append(f"- 本周做了两轮 Day2 refine，对同一批配置（{', '.join(cfgs)}）做复跑。")
    lines.append(f"- 第一轮最优是 `{old_best_cfg}`，第二轮最优切换为 `{new_best_cfg}`。")
    lines.append("- 从结果看，参数最优点发生了迁移，说明当前仍处在不够稳定的区间，后续需要多 seed 进一步确认。")
    lines.append("")
    lines.append("## 二、数据对比（两轮 Day2）")
    lines.append("")
    lines.append("| config | Round1 best mIoU | Round2 best mIoU | Delta | Round1 F1 | Round2 F1 | Round1 Acc | Round2 Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cfg, o, n, d in rows:
        lines.append(
            f"| {cfg} | {o.miou:.4f} | {n.miou:.4f} | {d:+.4f} | {o.f1:.4f} | {n.f1:.4f} | {o.acc:.2f} | {n.acc:.2f} |"
        )

    lines.append("")
    lines.append("## 三、推理图对比")
    lines.append("")
    lines.append("图里从左到右依次是：RGB、GT、第一轮最优模型、第二轮最优模型。")
    lines.append("")
    for vn in visual_list:
        lines.append(f"### {vn}")
        lines.append("")
        lines.append(f"![{vn}](weekly_report_assets/{vn})")
        lines.append("")

    lines.append("## 四、原因总结")
    lines.append("")
    lines.append("1. 这两轮用的是同一套超参数候选，但最优配置换了位置，说明训练对初始化与采样顺序还是敏感。")
    lines.append("2. 第二轮里 c03 拉高很明显，但 c05/c06 有回落，这不是单方向整体提升，更像是最优点在漂移。")
    lines.append("3. 目前单看单种子容易把偶然峰值当成稳定结论，下一步必须用多 seed 把方向压实。")
    lines.append("")
    lines.append("## 五、下周动作")
    lines.append("")
    lines.append("1. 先围绕第二轮最优和次优配置做 2-3 个 seed 的复验，优先看均值和方差。")
    lines.append("2. 保留一组回退配置做兜底，避免主线参数在跨天复跑里继续漂移。")
    lines.append("3. 继续补边界细节可视化，重点看道路-建筑贴边和窄边界断裂问题。")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"[Report] markdown: {output_md}")
    print(f"[Report] assets dir: {assets_dir}")


if __name__ == "__main__":
    main()
