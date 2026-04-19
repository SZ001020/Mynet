#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BestRow:
    run_id: str
    group: str
    best_miou: float
    best_f1: float
    best_acc: float
    epoch: int
    loss_mode: str
    lambda_adv: float
    lambda_bdy: float
    lambda_obj: float


def parse_args():
    p = argparse.ArgumentParser(description="Generate week5 day5 summary_t2_v3 and failure case note.")
    p.add_argument("--run-root", required=True, help="week5 run root, e.g. autodl-tmp/runs/week5/20260415_161039")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def group_name_from_run_id(run_id: str) -> str:
    mapping = {
        "source_only_seed": "source_only",
        "uda_seed": "uda",
        "uda_bdy_seed": "uda_bdy",
        "uda_obj_seed": "uda_obj",
        "uda_bdy_obj_seed": "uda_bdy_obj",
    }
    for k, v in mapping.items():
        if run_id.startswith(k):
            return v
    return run_id


def read_best(csv_path: str, run_id: str) -> Optional[BestRow]:
    if not os.path.isfile(csv_path):
        return None

    best = None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                miou = float(row.get("target_mean_miou", "nan"))
            except Exception:
                continue
            if math.isnan(miou):
                continue
            if best is None or miou > best.best_miou:
                best = BestRow(
                    run_id=run_id,
                    group=group_name_from_run_id(run_id),
                    best_miou=miou,
                    best_f1=float(row.get("target_mean_f1", "nan")),
                    best_acc=float(row.get("target_total_acc", "nan")),
                    epoch=int(row.get("epoch", 0)),
                    loss_mode=row.get("loss_mode", "NA"),
                    lambda_adv=float(row.get("lambda_adv", "nan")),
                    lambda_bdy=float(row.get("lambda_bdy", "nan")),
                    lambda_obj=float(row.get("lambda_obj", "nan")),
                )
    return best


def parse_day4_tile_table(md_path: str) -> List[Tuple[str, float, float, float]]:
    if not os.path.isfile(md_path):
        return []
    rows = []
    pat = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$")
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            tile = m.group(1).strip()
            if tile.lower() == "tile":
                continue
            rows.append((tile, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def write_file(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    run_root = os.path.abspath(args.run_root)

    csv_name = f"MFNet_week3_Vaihingen2Potsdam_seed{args.seed}.csv"
    candidates = [
        "source_only_seed42",
        "uda_seed42",
        "uda_bdy_seed42",
        "uda_obj_seed42",
        "uda_bdy_obj_seed42",
    ]

    best_rows: List[BestRow] = []
    missing = []
    for rid in candidates:
        csv_path = os.path.join(run_root, rid, csv_name)
        row = read_best(csv_path, rid)
        if row is None:
            missing.append(rid)
        else:
            best_rows.append(row)

    if not best_rows:
        raise RuntimeError("No valid week5 CSV rows found.")

    best_rows.sort(key=lambda x: x.best_miou, reverse=True)
    best_run = best_rows[0]

    by_run = {r.run_id: r for r in best_rows}
    uda = by_run.get("uda_seed42")
    src = by_run.get("source_only_seed42")
    uda_bdy = by_run.get("uda_bdy_seed42")
    uda_obj = by_run.get("uda_obj_seed42")
    uda_bdy_obj = by_run.get("uda_bdy_obj_seed42")

    # summary_t2_v3
    t2 = []
    t2.append("# T2-v3 Week5 主结果汇总")
    t2.append("")
    t2.append("## 结果总表（seed42）")
    t2.append("")
    t2.append("| rank | run_id | group | best_mIoU | best_F1 | best_Acc | epoch | loss_mode |")
    t2.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(best_rows, start=1):
        t2.append(
            f"| {i} | {r.run_id} | {r.group} | {r.best_miou:.4f} | {r.best_f1:.4f} | {r.best_acc:.2f} | {r.epoch} | {r.loss_mode} |"
        )

    t2.append("")
    t2.append("## 关键对比（相对 +UDA）")
    t2.append("")
    if uda is not None:
        def d(v: Optional[BestRow]) -> str:
            if v is None:
                return "N/A"
            return f"{(v.best_miou - uda.best_miou):+.4f}"
        t2.append(f"- +UDA+BDY vs +UDA: {d(uda_bdy)}")
        t2.append(f"- +UDA+OBJ vs +UDA: {d(uda_obj)}")
        t2.append(f"- +UDA+BDY+OBJ vs +UDA: {d(uda_bdy_obj)}")
    else:
        t2.append("- +UDA 组缺失，无法做相对增益比较。")

    t2.append("")
    t2.append("## 结论")
    t2.append("")
    t2.append(f"- 当前单 seed 最优为 {best_run.run_id}（mIoU={best_run.best_miou:.4f}）。")
    if src is not None and uda is not None:
        t2.append(f"- +UDA 相对 Source-only 提升 {(uda.best_miou - src.best_miou):+.4f}。")
    t2.append("- 该表可作为 Day5-5 的 T2-v3 主表提交版本。")
    if missing:
        t2.append(f"- 缺失组: {', '.join(missing)}")

    t2_path = os.path.join(run_root, "summary_t2_v3.md")
    write_file(t2_path, t2)

    # failure case note
    day4_path = os.path.join(run_root, "week5_day4_visual_summary.md")
    tile_rows = parse_day4_tile_table(day4_path)
    weakest = None
    if tile_rows:
        # sort by pseudo quality ascending, then boundary response descending
        weakest = sorted(tile_rows, key=lambda x: (x[2], -x[3]))[0]

    fc = []
    fc.append("# Week5 失败案例说明（Day5-5）")
    fc.append("")
    fc.append("## 结构组表现回落点")
    fc.append("")
    if uda is not None:
        for name, row in [("+UDA+BDY", uda_bdy), ("+UDA+BDY+OBJ", uda_bdy_obj)]:
            if row is None:
                fc.append(f"- {name}: 结果缺失。")
            else:
                fc.append(f"- {name}: 相对 +UDA 下降 {row.best_miou - uda.best_miou:+.4f}（{row.best_miou:.4f} vs {uda.best_miou:.4f}）。")
    else:
        fc.append("- +UDA 结果缺失，无法判断结构组回落幅度。")

    fc.append("")
    fc.append("## 典型困难样本（来自 F5-v1）")
    fc.append("")
    if weakest is not None:
        tile, conf, pq, br = weakest
        fc.append(f"- 最弱 tile: {tile}")
        fc.append(f"- mean_conf={conf:.4f}, mean_pseudo_quality={pq:.4f}, mean_boundary_response={br:.4f}")
        fc.append(f"- 对应图: pseudo_f5_v1/pseudo_tile_{tile}_pUDApOBJ.png")
    else:
        fc.append("- 未解析到 Day4 伪标签统计表，建议手动补充困难样本说明。")

    fc.append("")
    fc.append("## 原因归纳")
    fc.append("")
    fc.append("1. 结构项在当前权重下可能引入了额外噪声监督，边界区域更容易放大误差。")
    fc.append("2. 单 seed 结果对初始化和数据顺序敏感，局部峰值可能掩盖真实稳定性。")
    fc.append("3. 困难 tile 的边界响应偏高时，常伴随伪标签质量下降，需要更强置信门控。")

    fc.append("")
    fc.append("## 后续修正建议")
    fc.append("")
    fc.append("1. 优先补 2-seed 或 3-seed 统计，按 mean±std 决定是否保留结构项。")
    fc.append("2. 对 +BDY 与 +BDY+OBJ 分别下调 lambda_bdy/lambda_obj，做小网格复验。")
    fc.append("3. 在困难 tile 上增加可视化抽样，持续跟踪边界处误分模式。")

    fc_path = os.path.join(run_root, "week5_failure_cases.md")
    write_file(fc_path, fc)

    print(f"[Day5] generated: {t2_path}")
    print(f"[Day5] generated: {fc_path}")


if __name__ == "__main__":
    main()
