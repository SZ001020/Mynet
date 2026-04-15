#!/usr/bin/env python3
"""Generate Week4 F4-v3 stability curves from existing run CSV files.

Output:
- /root/Mynet/autodl-tmp/runs/week4_stabilization/f4_v3_week4_stability_curves.png
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class RunCurve:
    label: str
    day: str
    csv_path: str
    epochs: List[int]
    miou_epochs: List[int]
    miou_vals: List[float]
    adv_total: List[float]


def _to_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def load_curve(label: str, day: str, csv_path: str) -> RunCurve:
    epochs: List[int] = []
    miou_epochs: List[int] = []
    miou_vals: List[float] = []
    adv_total: List[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            feat = _to_float(row.get("train_adv_feat_loss", "nan"))
            disc = _to_float(row.get("train_adv_disc_loss", "nan"))
            miou = _to_float(row.get("target_mean_miou", "nan"))

            epochs.append(epoch)
            adv_total.append(feat + disc)
            if not math.isnan(miou):
                miou_epochs.append(epoch)
                miou_vals.append(miou)

    return RunCurve(
        label=label,
        day=day,
        csv_path=csv_path,
        epochs=epochs,
        miou_epochs=miou_epochs,
        miou_vals=miou_vals,
        adv_total=adv_total,
    )


def main() -> None:
    base = "/root/Mynet/autodl-tmp/runs/week4_stabilization"

    run_specs: List[Dict[str, str]] = [
        {
            "label": "Day1-c05",
            "day": "Day1",
            "csv": os.path.join(base, "day1_grid/20260408_205539/uda_grid/c05_l0.0030_g1.0_a0.0010/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
        {
            "label": "Day1-c06",
            "day": "Day1",
            "csv": os.path.join(base, "day1_grid/20260408_205539/uda_grid/c06_l0.0020_g1.5_a0.0015/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
        {
            "label": "Day1-c03",
            "day": "Day1",
            "csv": os.path.join(base, "day1_grid/20260408_205539/uda_grid/c03_l0.0010_g1.0_a0.0010/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
        {
            "label": "Day2-c06",
            "day": "Day2",
            "csv": os.path.join(base, "day2_refine/20260412_002225/c06_l0.0020_g1.5_a0.0015/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
        {
            "label": "Day2-c05",
            "day": "Day2",
            "csv": os.path.join(base, "day2_refine/20260412_002225/c05_l0.0030_g1.0_a0.0010/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
        {
            "label": "Day2-c03",
            "day": "Day2",
            "csv": os.path.join(base, "day2_refine/20260412_002225/c03_l0.0010_g1.0_a0.0010/MFNet_week3_Vaihingen2Potsdam_seed42.csv"),
        },
    ]

    curves: List[RunCurve] = []
    for spec in run_specs:
        if not os.path.isfile(spec["csv"]):
            raise FileNotFoundError(spec["csv"])
        curves.append(load_curve(spec["label"], spec["day"], spec["csv"]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for c in curves:
        linestyle = "--" if c.day == "Day1" else "-"
        ax1.plot(c.miou_epochs, c.miou_vals, marker="o", linewidth=2.0, linestyle=linestyle, label=c.label)

    for c in curves:
        linestyle = "--" if c.day == "Day1" else "-"
        ax2.plot(c.epochs, c.adv_total, linewidth=1.8, linestyle=linestyle, label=c.label)

    ax1.set_title("F4-v3 Week4 Stabilization: target mIoU vs epoch")
    ax1.set_ylabel("target mean mIoU")
    ax1.set_ylim(0.32, 0.58)

    ax2.set_title("F4-v3 Week4 Stabilization: adv loss (feat + disc) vs epoch")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("adv total loss")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(base, "f4_v3_week4_stability_curves.png")
    fig.savefig(out_path, dpi=180)
    print(out_path)


if __name__ == "__main__":
    main()
