import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv_records(csv_path):
    records = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["epoch"] = int(row.get("epoch", 0))
            row["iter"] = int(row.get("iter", 0))
            row["train_loss"] = _safe_float(row.get("train_loss"))
            row["val_metric"] = _safe_float(row.get("val_metric"))
            row["best_val_metric"] = _safe_float(row.get("best_val_metric"))
            row["lr"] = _safe_float(row.get("lr"))
            records.append(row)
    return records


def summarize_latest(records):
    if not records:
        return None
    best_row = None
    for r in records:
        if r["val_metric"] is None:
            continue
        if best_row is None or r["val_metric"] > best_row["val_metric"]:
            best_row = r
    return best_row


def plot_val_curves(grouped_records, output_path):
    plt.figure(figsize=(9, 5))
    plotted = False
    for name, records in sorted(grouped_records.items()):
        xs = []
        ys = []
        for r in records:
            if r["val_metric"] is None:
                continue
            xs.append(r["epoch"])
            ys.append(r["val_metric"])
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.2, label=name)
            plotted = True

    if not plotted:
        return False

    plt.title("Week-1 Baseline Validation Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def write_markdown(summary_rows, md_path, curve_rel_path):
    lines = []
    lines.append("# Week-1 Baseline Summary")
    lines.append("")
    lines.append("## Baseline Table")
    lines.append("")
    lines.append("| Model | Dataset | Seed | Best Epoch | Best Val Metric | LR | Batch | Window | Stride | CSV |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---|")

    if not summary_rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    else:
        for row in summary_rows:
            lines.append(
                "| {model} | {dataset} | {seed} | {epoch} | {metric:.4f} | {lr:.6f} | {batch} | {window} | {stride} | {csv} |".format(
                    model=row["model"],
                    dataset=row["dataset"],
                    seed=row["seed"],
                    epoch=row["epoch"],
                    metric=row["metric"],
                    lr=row["lr"],
                    batch=row["batch"],
                    window=row["window"],
                    stride=row["stride"],
                    csv=row["csv_name"],
                )
            )

    lines.append("")
    lines.append("## Curves")
    lines.append("")
    lines.append("![Validation Curves]({})".format(curve_rel_path))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Validation metric uses each model's current evaluation function output.")
    lines.append("- If you need strict mIoU/F1/Boundary-F1 alignment, add unified evaluator on saved prediction maps next.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    log_dir = os.environ.get("SSRS_LOG_DIR", "/root/SSRS/runs/week1_baseline")
    output_dir = os.environ.get("SSRS_SUMMARY_DIR", "/root/SSRS/runs/week1_baseline")

    os.makedirs(output_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    grouped = defaultdict(list)
    summary_rows = []

    for p in csv_paths:
        records = read_csv_records(p)
        if not records:
            continue

        key = "{}-{}-seed{}".format(
            records[0].get("model", "Unknown"),
            records[0].get("dataset", "Unknown"),
            records[0].get("seed", "NA"),
        )
        grouped[key].extend(records)

        best = summarize_latest(records)
        if best is None:
            continue

        summary_rows.append(
            {
                "model": best.get("model", "Unknown"),
                "dataset": best.get("dataset", "Unknown"),
                "seed": best.get("seed", "NA"),
                "epoch": best.get("epoch", 0),
                "metric": best.get("val_metric", 0.0) or 0.0,
                "lr": best.get("lr", 0.0) or 0.0,
                "batch": best.get("batch_size", "NA"),
                "window": best.get("window_size", "NA"),
                "stride": best.get("stride", "NA"),
                "csv_name": os.path.basename(p),
            }
        )

    curve_path = os.path.join(output_dir, "week1_val_curves.png")
    plotted = plot_val_curves(grouped, curve_path)
    if not plotted:
        # Create an empty placeholder image to keep markdown link stable.
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No curve data found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=160)
        plt.close()

    md_path = os.path.join(output_dir, "week1_baseline_summary.md")
    write_markdown(summary_rows, md_path, os.path.basename(curve_path))

    print("Summary markdown:", md_path)
    print("Curve image:", curve_path)


if __name__ == "__main__":
    main()
