import csv
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_csv_records(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["epoch"] = int(row.get("epoch", 0) or 0)
            row["train_loss"] = _safe_float(row.get("train_loss"))
            row["val_metric"] = _safe_float(row.get("val_metric"))
            row["mean_f1"] = _safe_float(row.get("mean_f1"))
            row["kappa"] = _safe_float(row.get("kappa"))
            row["mean_miou"] = _safe_float(row.get("mean_miou"))
            row["train_loss_ce"] = _safe_float(row.get("train_loss_ce"))
            row["train_loss_boundary"] = _safe_float(row.get("train_loss_boundary"))
            row["train_loss_object"] = _safe_float(row.get("train_loss_object"))
            rows.append(row)
    return rows


def _best_row(rows):
    best = None
    for r in rows:
        if r["val_metric"] is None:
            continue
        if best is None or r["val_metric"] > best["val_metric"]:
            best = r
    return best


def _latest_timestamp_dir(parent_dir):
    subdirs = [
        d for d in glob.glob(os.path.join(parent_dir, "*"))
        if os.path.isdir(d)
    ]
    if not subdirs:
        return None

    def _key(path):
        name = os.path.basename(path)
        try:
            return datetime.strptime(name, "%Y%m%d_%H%M%S")
        except ValueError:
            return datetime.min

    subdirs.sort(key=_key, reverse=True)
    return subdirs[0]


def _find_fallback_baseline_csv(week2_root_parent, exclude_dir):
    pattern = os.path.join(week2_root_parent, "*", "baseline_seg", "*.csv")
    candidates = []
    for p in glob.glob(pattern):
        run_dir = os.path.dirname(os.path.dirname(p))
        if os.path.abspath(run_dir) == os.path.abspath(exclude_dir):
            continue
        if os.path.getsize(p) <= 0:
            continue
        candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def _plot_curves(run_rows_map, output_path, metric_key, title, ylabel):
    plt.figure(figsize=(9, 5))
    plotted = False
    for run_name, rows in run_rows_map.items():
        xs, ys = [], []
        for r in rows:
            y = r.get(metric_key)
            if y is None:
                continue
            xs.append(r["epoch"])
            ys.append(y)
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.2, label=run_name)
            plotted = True

    if not plotted:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, f"No {metric_key} data found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    week2_root_parent = os.environ.get(
        "SSRS_WEEK2_ROOT",
        os.path.join(repo_root, "runs", "week2_same_domain"),
    )
    target_dir = os.environ.get("SSRS_WEEK2_DIR", "")
    if not target_dir:
        target_dir = _latest_timestamp_dir(week2_root_parent)
    if not target_dir or not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Week2 run directory not found: {target_dir}")

    baseline_csv = glob.glob(os.path.join(target_dir, "baseline_seg", "*.csv"))
    struct_csv = glob.glob(os.path.join(target_dir, "struct_seg_bdy_obj", "*.csv"))

    run_rows = {}
    baseline_source = ""
    if baseline_csv:
        run_rows["Baseline"] = _read_csv_records(baseline_csv[0])
        baseline_source = baseline_csv[0]
    else:
        fallback = _find_fallback_baseline_csv(week2_root_parent, target_dir)
        if fallback:
            run_rows["Baseline"] = _read_csv_records(fallback)
            baseline_source = fallback

    if struct_csv:
        run_rows["Structured"] = _read_csv_records(struct_csv[0])

    summary_path = os.path.join(target_dir, "week2_same_domain_summary.md")
    val_curve_path = os.path.join(target_dir, "week2_val_curves.png")
    loss_curve_path = os.path.join(target_dir, "week2_loss_curves.png")

    _plot_curves(run_rows, val_curve_path, "val_metric", "Week-2 Same-Domain Validation Curves", "Validation Metric")
    _plot_curves(run_rows, loss_curve_path, "train_loss", "Week-2 Same-Domain Training Loss Curves", "Train Loss")

    base_best = _best_row(run_rows.get("Baseline", [])) if "Baseline" in run_rows else None
    struct_best = _best_row(run_rows.get("Structured", [])) if "Structured" in run_rows else None

    def fmt(v, nd=4):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = []
    lines.append("# Week-2 Same-Domain Summary")
    lines.append("")
    lines.append("## Comparison Table")
    lines.append("")
    lines.append("| Run | Loss Mode | Best Epoch | Best Val Metric | mean F1 | Kappa | mean MIoU | Train Loss | CE Loss | Boundary Loss | Object Loss | Batch | Window |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    if base_best is not None:
        lines.append(
            "| Baseline | {loss_mode} | {epoch} | {val_metric} | {mean_f1} | {kappa} | {mean_miou} | {train_loss} | {ce} | {bdy} | {obj} | {batch} | {window} |".format(
                loss_mode=fmt(base_best.get("loss_mode", "SEG")),
                epoch=fmt(base_best.get("epoch"), 0),
                val_metric=fmt(base_best.get("val_metric")),
                mean_f1=fmt(base_best.get("mean_f1")),
                kappa=fmt(base_best.get("kappa")),
                mean_miou=fmt(base_best.get("mean_miou")),
                train_loss=fmt(base_best.get("train_loss")),
                ce=fmt(base_best.get("train_loss_ce")),
                bdy=fmt(base_best.get("train_loss_boundary")),
                obj=fmt(base_best.get("train_loss_object")),
                batch=fmt(base_best.get("batch_size")),
                window=fmt(base_best.get("window_size")),
            )
        )
    else:
        lines.append("| Baseline | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    if struct_best is not None:
        lines.append(
            "| Structured | {loss_mode} | {epoch} | {val_metric} | {mean_f1} | {kappa} | {mean_miou} | {train_loss} | {ce} | {bdy} | {obj} | {batch} | {window} |".format(
                loss_mode=fmt(struct_best.get("loss_mode", "SEG+BDY+OBJ")),
                epoch=fmt(struct_best.get("epoch"), 0),
                val_metric=fmt(struct_best.get("val_metric")),
                mean_f1=fmt(struct_best.get("mean_f1")),
                kappa=fmt(struct_best.get("kappa")),
                mean_miou=fmt(struct_best.get("mean_miou")),
                train_loss=fmt(struct_best.get("train_loss")),
                ce=fmt(struct_best.get("train_loss_ce")),
                bdy=fmt(struct_best.get("train_loss_boundary")),
                obj=fmt(struct_best.get("train_loss_object")),
                batch=fmt(struct_best.get("batch_size")),
                window=fmt(struct_best.get("window_size")),
            )
        )
    else:
        lines.append("| Structured | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.append("")
    lines.append("## Delta")
    lines.append("")
    if base_best and struct_best:
        delta = (struct_best.get("val_metric") or 0.0) - (base_best.get("val_metric") or 0.0)
        lines.append(f"- Delta(Structured - Baseline) on best val metric: {delta:.4f}")
    else:
        lines.append("- Delta unavailable because one of the runs is missing.")

    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- Baseline CSV source: {baseline_source if baseline_source else 'N/A'}")
    lines.append(f"- Structured CSV source: {struct_csv[0] if struct_csv else 'N/A'}")

    lines.append("")
    lines.append("## Curves")
    lines.append("")
    lines.append("![Week2 Validation Curves](week2_val_curves.png)")
    lines.append("")
    lines.append("![Week2 Training Loss Curves](week2_loss_curves.png)")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Week2 summary:", summary_path)
    print("Week2 val curve:", val_curve_path)
    print("Week2 loss curve:", loss_curve_path)


if __name__ == "__main__":
    main()
