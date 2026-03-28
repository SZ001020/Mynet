import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_model_name(name):
    s = (name or "").strip().lower()
    if "ftrans" in s:
        return "FTransUNet"
    if "asmf" in s:
        return "ASMFNet"
    if "unetformer" in s or "mfnet" in s:
        return "MFNet"
    return name or "Unknown"


def _parse_last_metrics_block(log_path):
    if not os.path.isfile(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    latest = None
    for i, line in enumerate(lines):
        if not line.startswith("Total accuracy"):
            continue

        block = {
            "total_acc": _safe_float(line.split(":", 1)[-1].strip()),
            "mean_f1": None,
            "kappa": None,
            "mean_miou": None,
            "roads_f1": None,
            "buildings_f1": None,
            "low_veg_f1": None,
            "trees_f1": None,
            "cars_f1": None,
            "clutter_f1": None,
        }

        in_f1_section = False
        for j in range(i + 1, min(i + 80, len(lines))):
            s = lines[j]
            if s.startswith("F1Score"):
                in_f1_section = True
                continue
            if s.startswith("mean F1Score"):
                block["mean_f1"] = _safe_float(s.split(":", 1)[-1].strip())
                in_f1_section = False
                continue
            if s.startswith("Kappa"):
                block["kappa"] = _safe_float(s.split(":", 1)[-1].strip())
                continue
            if s.startswith("mean MIoU"):
                block["mean_miou"] = _safe_float(s.split(":", 1)[-1].strip())
                break

            if in_f1_section and ":" in s:
                k, v = s.split(":", 1)
                k = k.strip().lower()
                vv = _safe_float(v.strip())
                if k == "roads":
                    block["roads_f1"] = vv
                elif k == "buildings":
                    block["buildings_f1"] = vv
                elif k in ("low veg.", "low veg"):
                    block["low_veg_f1"] = vv
                elif k == "trees":
                    block["trees_f1"] = vv
                elif k == "cars":
                    block["cars_f1"] = vv
                elif k == "clutter":
                    block["clutter_f1"] = vv

        latest = block

    return latest


def _discover_model_logs(base_runs_dir):
    log_map = {}
    for p in glob.glob(os.path.join(base_runs_dir, "**", "train.log"), recursive=True):
        lp = p.lower()
        model = None
        if "ftrans" in lp:
            model = "FTransUNet"
        elif "asmf" in lp:
            model = "ASMFNet"
        elif "mfnet" in lp:
            model = "MFNet"

        if model is None:
            continue

        if model not in log_map:
            log_map[model] = p
        else:
            if os.path.getmtime(p) > os.path.getmtime(log_map[model]):
                log_map[model] = p
    return log_map


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


def plot_loss_curves(grouped_records, output_path):
    plt.figure(figsize=(9, 5))
    plotted = False
    for name, records in sorted(grouped_records.items()):
        xs = []
        ys = []
        for r in records:
            if r["train_loss"] is None:
                continue
            xs.append(r["epoch"])
            ys.append(r["train_loss"])
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.2, label=name)
            plotted = True

    if not plotted:
        return False

    plt.title("Week-1 Baseline Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def write_markdown(summary_rows, md_path, curve_rel_path, loss_curve_rel_path):
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
    lines.append("![Training Loss Curves]({})".format(loss_curve_rel_path))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Validation metric uses each model's current evaluation function output.")
    lines.append("- If you need strict mIoU/F1/Boundary-F1 alignment, add unified evaluator on saved prediction maps next.")

    lines.append("")
    lines.append("## Detailed Metrics (From Train Logs)")
    lines.append("")
    lines.append("| Model | Total Acc | mean F1 | Kappa | mean MIoU | roads F1 | buildings F1 | low veg F1 | trees F1 | cars F1 | clutter F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    has_detailed = False
    for row in summary_rows:
        m = row.get("detailed_metrics")
        if not m:
            continue
        has_detailed = True
        lines.append(
            "| {model} | {total_acc} | {mean_f1} | {kappa} | {mean_miou} | {roads_f1} | {buildings_f1} | {low_veg_f1} | {trees_f1} | {cars_f1} | {clutter_f1} |".format(
                model=row["model"],
                total_acc=("{:.4f}".format(m["total_acc"]) if m.get("total_acc") is not None else "N/A"),
                mean_f1=("{:.4f}".format(m["mean_f1"]) if m.get("mean_f1") is not None else "N/A"),
                kappa=("{:.4f}".format(m["kappa"]) if m.get("kappa") is not None else "N/A"),
                mean_miou=("{:.4f}".format(m["mean_miou"]) if m.get("mean_miou") is not None else "N/A"),
                roads_f1=("{:.4f}".format(m["roads_f1"]) if m.get("roads_f1") is not None else "N/A"),
                buildings_f1=("{:.4f}".format(m["buildings_f1"]) if m.get("buildings_f1") is not None else "N/A"),
                low_veg_f1=("{:.4f}".format(m["low_veg_f1"]) if m.get("low_veg_f1") is not None else "N/A"),
                trees_f1=("{:.4f}".format(m["trees_f1"]) if m.get("trees_f1") is not None else "N/A"),
                cars_f1=("{:.4f}".format(m["cars_f1"]) if m.get("cars_f1") is not None else "N/A"),
                clutter_f1=("{:.4f}".format(m["clutter_f1"]) if m.get("clutter_f1") is not None else "N/A"),
            )
        )

    if not has_detailed:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    default_runs_dir = os.path.join(repo_root, "runs")
    default_week1_dir = os.path.join(default_runs_dir, "week1_baseline")

    log_dir = os.environ.get("SSRS_LOG_DIR", default_week1_dir)
    output_dir = os.environ.get("SSRS_SUMMARY_DIR", default_week1_dir)

    os.makedirs(output_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    runs_root = os.environ.get("SSRS_RUNS_ROOT", default_runs_dir)
    model_logs = _discover_model_logs(runs_root)
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

        model_name = best.get("model", "Unknown")
        normalized_model = _normalize_model_name(model_name)
        detailed = None
        if normalized_model in model_logs:
            detailed = _parse_last_metrics_block(model_logs[normalized_model])

        summary_rows.append(
            {
                "model": model_name,
                "dataset": best.get("dataset", "Unknown"),
                "seed": best.get("seed", "NA"),
                "epoch": best.get("epoch", 0),
                "metric": best.get("val_metric", 0.0) or 0.0,
                "lr": best.get("lr", 0.0) or 0.0,
                "batch": best.get("batch_size", "NA"),
                "window": best.get("window_size", "NA"),
                "stride": best.get("stride", "NA"),
                "csv_name": os.path.basename(p),
                "detailed_metrics": detailed,
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

    loss_curve_path = os.path.join(output_dir, "week1_loss_curves.png")
    plotted_loss = plot_loss_curves(grouped, loss_curve_path)
    if not plotted_loss:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No loss data found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=160)
        plt.close()

    md_path = os.path.join(output_dir, "week1_baseline_summary.md")
    write_markdown(
        summary_rows,
        md_path,
        os.path.basename(curve_path),
        os.path.basename(loss_curve_path),
    )

    print("Summary markdown:", md_path)
    print("Curve image:", curve_path)


if __name__ == "__main__":
    main()
