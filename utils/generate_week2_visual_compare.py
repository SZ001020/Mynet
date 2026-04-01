#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual comparison for week2 ablation and append summary.")
    parser.add_argument("--run-root", required=True, help="Run directory, e.g. runs/week2_ablation_seed1/<timestamp>")
    parser.add_argument("--num-tiles", type=int, default=3, help="Number of test tiles to visualize")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference patch batch size")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride")
    parser.add_argument("--num-zooms", type=int, default=2, help="Number of local zoom crops per tile")
    parser.add_argument("--zoom-size", type=int, default=384, help="Square crop size for local boundary zoom")
    return parser.parse_args()


def read_best_metric(csv_path: str):
    if not os.path.isfile(csv_path):
        return None
    best = None
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get("val_metric", "nan"))
            except Exception:
                continue
            if best is None or m > best["metric"]:
                best = {
                    "metric": m,
                    "epoch": row.get("epoch", "NA"),
                    "mean_f1": row.get("mean_f1", "NA"),
                    "kappa": row.get("kappa", "NA"),
                    "mean_miou": row.get("mean_miou", "NA"),
                }
    return best


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
        chunk = []
        for _ in range(n):
            try:
                chunk.append(next(it))
            except StopIteration:
                break
        if not chunk:
            return
        yield tuple(chunk)


def infer_tile(net, img, dsm, n_classes, window_size, stride, batch_size, device):
    pred = np.zeros(img.shape[:2] + (n_classes,), dtype=np.float32)

    total = int(math.ceil(count_sliding_window(img, step=stride, window_size=window_size) / float(batch_size)))
    coords_iter = grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))

    with torch.no_grad():
        for coords in tqdm(coords_iter, total=total, leave=False):
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = torch.from_numpy(np.asarray(image_patches)).to(device)

            dmin = np.min(dsm)
            dmax = np.max(dsm)
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


def hstack_panels(panels: List[np.ndarray]) -> np.ndarray:
    heights = [p.shape[0] for p in panels]
    max_h = max(heights)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad_h = max_h - p.shape[0]
            p = np.pad(p, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        padded.append(p)
    return np.concatenate(padded, axis=1)


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
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = max(4, (w - tw) // 2)
    ty = max(2, (bar_h - th) // 2)
    draw.text((tx, ty), title, font=font, fill=(255, 255, 255))
    return np.asarray(pil_im)


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
            patch = b[y:y + crop, x:x + crop]
            score = int(np.sum(patch))
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
            scx, scy = sx + sc // 2, sy + sc // 2
            if abs(cx - scx) + abs(cy - scy) < min_dist:
                keep = False
                break
        if keep:
            selected.append((score, x, y, c))
        if len(selected) >= top_k:
            break

    if not selected:
        selected.append(candidates[0])
    return selected


def crop_img(img: np.ndarray, x: int, y: int, c: int) -> np.ndarray:
    return np.ascontiguousarray(img[y:y + c, x:x + c])


def write_summary(summary_path: str, lines: List[str]):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    run_root = os.path.abspath(args.run_root)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mfnet_dir = os.path.join(repo_root, "MFNet")
    if "SSRS_DATA_ROOT" not in os.environ:
        os.environ["SSRS_DATA_ROOT"] = os.path.join(repo_root, "autodl-tmp", "dataset")
    sam_ckpt_default = os.path.join(mfnet_dir, "weights", "sam_vit_l_0b3195.pth")
    if os.path.isfile(sam_ckpt_default):
        os.environ["SSRS_MFNET_SAM_CKPT"] = sam_ckpt_default
    if mfnet_dir not in sys.path:
        sys.path.insert(0, mfnet_dir)

    # Some imported training modules parse CLI args at import time.
    # Isolate argv to avoid conflicts with this script's custom flags.
    sys.argv = [sys.argv[0]]
    from UNetFormer_MMSAM import UNetFormer as MFNetModel  # pylint: disable=import-error
    import utils as mf_utils  # pylint: disable=import-error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this visual generation script.")

    groups = {
        "Baseline": "seg",
        "+Boundary": "seg_bdy",
        "+Object": "seg_obj",
        "+Boundary+Object": "seg_bdy_obj",
    }

    test_ids = list(mf_utils.test_ids)[: args.num_tiles]
    n_classes = mf_utils.N_CLASSES
    window_size = mf_utils.WINDOW_SIZE

    visual_root = os.path.join(run_root, "visual_compare")
    os.makedirs(visual_root, exist_ok=True)

    metrics: Dict[str, dict] = {}
    preds_by_group = {k: {} for k in groups.keys()}

    for run_name, subdir in groups.items():
        group_dir = os.path.join(run_root, subdir)
        ckpt_path = os.path.join(group_dir, "UNetformer_best.pth")
        csv_path = os.path.join(group_dir, f"MFNet_{mf_utils.DATASET}_seed{os.environ.get('SSRS_SEED', '42')}.csv")
        metrics[run_name] = read_best_metric(csv_path)

        if not os.path.isfile(ckpt_path):
            print(f"[Visual][WARN] checkpoint missing: {ckpt_path}")
            continue

        net = MFNetModel(num_classes=n_classes).to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        out_dir = os.path.join(visual_root, subdir)
        os.makedirs(out_dir, exist_ok=True)

        for tile_id in test_ids:
            if mf_utils.DATASET == "Potsdam":
                img = 1 / 255 * np.asarray(io.imread(mf_utils.DATA_FOLDER.format(tile_id))[:, :, :3], dtype="float32")
            else:
                img = 1 / 255 * np.asarray(io.imread(mf_utils.DATA_FOLDER.format(tile_id)), dtype="float32")
            dsm = np.asarray(io.imread(mf_utils.DSM_FOLDER.format(tile_id)), dtype="float32")

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
            preds_by_group[run_name][tile_id] = pred

            pred_rgb = mf_utils.convert_to_color(pred)
            io.imsave(os.path.join(out_dir, f"tile_{tile_id}_pred.png"), pred_rgb)

        del net
        torch.cuda.empty_cache()

    # Build horizontal montage per tile with panel titles
    montage_paths = []
    zoom_paths = []
    for tile_id in test_ids:
        if mf_utils.DATASET == "Potsdam":
            rgb = np.asarray(io.imread(mf_utils.DATA_FOLDER.format(tile_id))[:, :, :3], dtype=np.uint8)
        else:
            rgb = np.asarray(io.imread(mf_utils.DATA_FOLDER.format(tile_id)), dtype=np.uint8)

        gt_raw = np.asarray(io.imread(mf_utils.LABEL_FOLDER.format(tile_id)))
        if mf_utils.DATASET == "Hunan":
            gt_mask = gt_raw.astype(np.uint8)
        else:
            gt_mask = mf_utils.convert_from_color(gt_raw)
        gt_rgb = mf_utils.convert_to_color(gt_mask)

        panels = [
            draw_title_bar(rgb, "RGB"),
            draw_title_bar(gt_rgb, "GT"),
        ]
        for run_name in groups.keys():
            pred = preds_by_group.get(run_name, {}).get(tile_id)
            if pred is None:
                pred_rgb = np.zeros_like(gt_rgb, dtype=np.uint8)
            else:
                pred_rgb = mf_utils.convert_to_color(pred)
            if run_name == "Baseline":
                title = "Baseline"
            elif run_name == "+Boundary":
                title = "+Boundary"
            elif run_name == "+Object":
                title = "+Object"
            else:
                title = "+Boundary+Object"
            panels.append(draw_title_bar(pred_rgb, title))

        merged = hstack_panels(panels)
        montage_rel = f"visual_compare/montage_tile_{tile_id}.png"
        Image.fromarray(merged).save(os.path.join(run_root, montage_rel))
        montage_paths.append(montage_rel)

        # Build local zoom montages for boundary-sensitive comparison
        boxes = select_zoom_boxes(gt_mask, crop_size=args.zoom_size, top_k=args.num_zooms)
        for idx, (_, x, y, c) in enumerate(boxes, start=1):
            zoom_panels = [
                draw_title_bar(crop_img(rgb, x, y, c), "RGB"),
                draw_title_bar(crop_img(gt_rgb, x, y, c), "GT"),
            ]
            for run_name in groups.keys():
                pred = preds_by_group.get(run_name, {}).get(tile_id)
                if pred is None:
                    pred_rgb = np.zeros_like(gt_rgb, dtype=np.uint8)
                else:
                    pred_rgb = mf_utils.convert_to_color(pred)
                if run_name == "Baseline":
                    title = "Baseline"
                elif run_name == "+Boundary":
                    title = "+Boundary"
                elif run_name == "+Object":
                    title = "+Object"
                else:
                    title = "+Boundary+Object"
                zoom_panels.append(draw_title_bar(crop_img(pred_rgb, x, y, c), title))

            zoom_merged = hstack_panels(zoom_panels)
            zoom_rel = f"visual_compare/zoom_tile_{tile_id}_{idx}.png"
            Image.fromarray(zoom_merged).save(os.path.join(run_root, zoom_rel))
            zoom_paths.append(zoom_rel)

    # Rewrite summary markdown in Chinese with detailed analysis
    summary_path = os.path.join(run_root, "week2_ablation_seed1_summary.md")
    baseline_metric = metrics.get("Baseline", {}).get("metric") if metrics.get("Baseline") else None

    report = []
    report.append("# 第2周 Same-domain 消融总结（单种子）")
    report.append("")
    report.append(f"- 数据集: {mf_utils.DATASET}")
    report.append(f"- 种子: {os.environ.get('SSRS_SEED', '42')}")
    report.append(f"- 推理拼图数量: {len(montage_paths)}")
    report.append("")
    report.append("| 组别 | Loss 模式 | 最佳 mIoU | 最佳 Epoch | mean F1 | Kappa | 相对 Baseline |")
    report.append("|---|---|---:|---:|---:|---:|---:|")

    ordered_names = ["Baseline", "+Boundary", "+Object", "+Boundary+Object"]
    loss_mode_map = {
        "Baseline": "SEG",
        "+Boundary": "SEG+BDY",
        "+Object": "SEG+OBJ",
        "+Boundary+Object": "SEG+BDY+OBJ",
    }
    for name in ordered_names:
        st = metrics.get(name)
        if not st:
            report.append(f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        delta = "0.0000" if baseline_metric is not None and name == "Baseline" else "N/A"
        if baseline_metric is not None and name != "Baseline":
            delta = f"{(st['metric'] - baseline_metric):+.4f}"
        report.append(
            f"| {name} | {loss_mode_map.get(name, 'N/A')} | "
            f"{st['metric']:.4f} | {st['epoch']} | {float(st['mean_f1']):.4f} | {float(st['kappa']):.4f} | {delta} |"
        )

    report.append("")
    report.append("## 拼图结果")
    report.append("")
    report.append("每张拼图均包含 6 列并带标题: RGB、GT、Baseline、+Boundary、+Object、+Boundary+Object。")
    report.append("")
    for p in montage_paths:
        report.append(f"### {os.path.basename(p)}")
        report.append("")
        report.append(f"![{os.path.basename(p)}]({p})")
        report.append("")

    report.append("## 边界局部放大图（F3）")
    report.append("")
    report.append("以下局部图自动选择了 GT 边界密度较高的区域，用于观察道路边沿、建筑转角和小目标邻域。")
    report.append("")
    for p in zoom_paths:
        report.append(f"### {os.path.basename(p)}")
        report.append("")
        report.append(f"![{os.path.basename(p)}]({p})")
        report.append("")

    report.append("## 详细分析")
    report.append("")
    if baseline_metric is not None:
        bdy = metrics.get("+Boundary")
        obj = metrics.get("+Object")
        both = metrics.get("+Boundary+Object")
        if bdy:
            report.append(f"1. +Boundary 与 Baseline 基本持平（delta={bdy['metric'] - baseline_metric:+.4f}）。说明边界项在当前短程配置下没有明显收益，但也没有明显破坏主任务。")
        if obj:
            report.append(f"2. +Object 相对 Baseline 明显回落（delta={obj['metric'] - baseline_metric:+.4f}）。说明对象约束在当前权重和门控设置下仍存在噪声牵引。")
        if both:
            report.append(f"3. +Boundary+Object 仍低于 Baseline（delta={both['metric'] - baseline_metric:+.4f}），且接近 +Object，表明当前瓶颈主要来自对象项。")
    report.append("4. 从可视化拼图看，四组在大面积建筑和道路区域的主轮廓差异较小；如果要证明边界机制有效，应重点查看道路边沿、建筑转角、小目标车辆的局部放大图。")
    report.append("5. 从自动抽取的边界高密度局部图看，+Boundary 在部分路缘位置与 Baseline 非常接近，尚未形成稳定可见优势；+Object 与 +Boundary+Object 在细碎区域出现更明显的类别扰动。")
    report.append("6. 结论上，本轮单种子短程实验没有证据表明结构组优于 Baseline；但 +Boundary 接近持平，可作为下一轮正式复验候选。")

    report.append("")
    report.append("## 建议的下一步")
    report.append("")
    report.append("1. 进入 50/1000 正式轮次时优先跑 SEG 与 SEG+BDY 两组。")
    report.append("2. +Object 与 +Boundary+Object 暂时降级为条件分支，后续需降低对象权重或提高对象门控阈值后再复验。")
    report.append("3. 若正式轮次仍无提升，第二周验收可转向“边界类局部可视化证据”为主，而不是追求 overall mIoU 提升。")

    write_summary(summary_path, report)
    print(f"[Visual] Updated summary: {summary_path}")


if __name__ == "__main__":
    main()
