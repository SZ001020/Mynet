#!/usr/bin/env python3
"""Summarize P13-G run metrics."""

from __future__ import annotations

import argparse
import glob
import json
import os


def fmt_metric(metrics, mode, key, default="-"):
    try:
        value = metrics[mode][key]
    except KeyError:
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="/root/autodl-tmp/runs")
    parser.add_argument("--pattern", default="plan13_g_g*_vaihingen_*")
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()

    by_experiment = {}
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_root, args.pattern))):
        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        config = json.load(open(config_path)) if os.path.exists(config_path) else {}
        exp = config.get("experiment", os.path.basename(run_dir))
        if not os.path.exists(metrics_path):
            if args.include_missing:
                rows.append([exp, "missing", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        metrics = json.load(open(metrics_path))
        by_experiment[exp] = [
            exp,
            config.get("select_mask", "-"),
            fmt_metric(metrics, "oracle", "avg_miou"),
            fmt_metric(metrics, "pred", "avg_miou"),
            fmt_metric(metrics, "oracle", "grass_to_tree_pct"),
            fmt_metric(metrics, "oracle", "tree_to_grass_pct"),
            fmt_metric(metrics, "pred", "grass_to_tree_pct"),
            fmt_metric(metrics, "pred", "tree_to_grass_pct"),
            fmt_metric(metrics, "pred", "veg_sum"),
            fmt_metric(metrics, "pred", "invariant_errors"),
        ]

    rows.extend(by_experiment[k] for k in sorted(by_experiment))

    headers = [
        "experiment", "best", "oracle mIoU", "pred mIoU",
        "oracle g->t%", "oracle t->g%", "pred g->t%", "pred t->g%",
        "pred veg sum", "pred inv",
    ]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
