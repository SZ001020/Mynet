#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_DATASET="${SSRS_DATASET:-Vaihingen}"
export SSRS_SEED="${SSRS_SEED:-42}"
export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE="${SSRS_BATCH_SIZE:-10}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"
export SSRS_PERF_MODE="${SSRS_PERF_MODE:-1}"
export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-16}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-1}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-4}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"
export SSRS_MFNET_MICRO_BS="${SSRS_MFNET_MICRO_BS:-2}"

# Debug/quick-run controls can be overridden externally.
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-1000}"
export SSRS_EPOCHS="${SSRS_EPOCHS:-50}"
export SSRS_SAVE_EPOCH="${SSRS_SAVE_EPOCH:-1}"
export SSRS_SAVE_BEST="${SSRS_SAVE_BEST:-1}"
export SSRS_SAVE_LAST="${SSRS_SAVE_LAST:-1}"
export SSRS_SAVE_INTERVAL="${SSRS_SAVE_INTERVAL:-0}"
export SSRS_EVAL_NUM_TILES="${SSRS_EVAL_NUM_TILES:-0}"

export SSRS_LAMBDA_BDY="${SSRS_LAMBDA_BDY:-0.1}"
export SSRS_LAMBDA_OBJ="${SSRS_LAMBDA_OBJ:-1.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
if [[ "$OMP_NUM_THREADS" -le 0 ]]; then
  export OMP_NUM_THREADS=16
fi
if [[ "$MKL_NUM_THREADS" -le 0 ]]; then
  export MKL_NUM_THREADS=16
fi

TS="$(date +%Y%m%d_%H%M%S)"
WEEK2_ROOT="${SSRS_WEEK2_OUT_DIR:-$ROOT_DIR/runs/week2_same_domain/$TS}"
BASELINE_DIR="$WEEK2_ROOT/baseline_seg"
STRUCT_DIR="$WEEK2_ROOT/struct_seg_bdy_obj"
mkdir -p "$STRUCT_DIR"
export WEEK2_ROOT

echo "[Week2] Root dir:      $ROOT_DIR"
echo "[Week2] Data root:      $SSRS_DATA_ROOT"
echo "[Week2] Dataset:        $SSRS_DATASET"
echo "[Week2] Seed:           $SSRS_SEED"
echo "[Week2] Output root:    $WEEK2_ROOT"
echo "[Week2] EPOCHS/STEPS:   $SSRS_EPOCHS/$SSRS_EPOCH_STEPS"
echo "[Week2] BATCH/WINDOW:   $SSRS_BATCH_SIZE/$SSRS_WINDOW_SIZE"
echo "[Week2] Lambda bdy/obj: $SSRS_LAMBDA_BDY/$SSRS_LAMBDA_OBJ"

echo "[Week2][Check] Running readiness checks..."
if [[ ! -d "$SSRS_DATA_ROOT" ]]; then
  echo "[Week2][FAIL] SSRS_DATA_ROOT not found: $SSRS_DATA_ROOT"
  exit 1
fi

if [[ ! -f "$MFNET_DIR/train.py" ]]; then
  echo "[Week2][FAIL] MFNet train.py not found: $MFNET_DIR/train.py"
  exit 1
fi

python - <<'PY'
import os
import sys
import torch

root = os.environ["SSRS_DATA_ROOT"]
dataset = os.environ.get("SSRS_DATASET", "Vaihingen")

if not torch.cuda.is_available():
    print("[Week2][FAIL] CUDA is not available.")
    sys.exit(1)

if dataset == "Vaihingen":
    sample = {
        "img": os.path.join(root, "Vaihingen/top/top_mosaic_09cm_area1.tif"),
        "dsm": os.path.join(root, "Vaihingen/dsm/dsm_09cm_matching_area1.tif"),
        "label": os.path.join(root, "Vaihingen/gts_for_participants/top_mosaic_09cm_area1.tif"),
        "eroded": os.path.join(root, "Vaihingen/gts_eroded_for_participants/top_mosaic_09cm_area1_noBoundary.tif"),
        "bdy": os.path.join(root, "Vaihingen/sam_boundary_merge/ISPRS_merge_1.tif"),
        "obj": os.path.join(root, "Vaihingen/V_merge/V_merge_1.tif"),
    }
elif dataset == "Potsdam":
    sample = {
        "img": os.path.join(root, "Potsdam/4_Ortho_RGBIR/top_potsdam_6_10_RGBIR.tif"),
        "dsm": os.path.join(root, "Potsdam/1_DSM_normalisation/dsm_potsdam_6_10_normalized_lastools.jpg"),
        "label": os.path.join(root, "Potsdam/5_Labels_for_participants/top_potsdam_6_10_label.tif"),
        "eroded": os.path.join(root, "Potsdam/5_Labels_for_participants_no_Boundary/top_potsdam_6_10_label_noBoundary.tif"),
        "bdy": os.path.join(root, "Potsdam/sam_boundary_merge/ISPRS_merge_6_10.tif"),
        "obj": os.path.join(root, "Potsdam/P_merge/P_merge_6_10.tif"),
    }
else:
    print(f"[Week2][WARN] Dataset '{dataset}' does not have built-in sample checks.")
    print("[Week2][PASS] CUDA available, continuing.")
    sys.exit(0)

missing_hard = [k for k in ("img", "dsm", "label", "eroded") if not os.path.isfile(sample[k])]
if missing_hard:
    print("[Week2][FAIL] Missing required files:")
    for k in missing_hard:
        print(f"  - {k}: {sample[k]}")
    sys.exit(1)

missing_soft = [k for k in ("bdy", "obj") if not os.path.isfile(sample[k])]
if missing_soft:
    print("[Week2][WARN] Missing boundary/object priors; structured run will degenerate to CE-heavy training:")
    for k in missing_soft:
        print(f"  - {k}: {sample[k]}")

print("[Week2][PASS] Readiness checks passed.")
PY

echo "[Week2] Run: MFNet structured only (SEG+BDY+OBJ)"
(
  cd "$MFNET_DIR"
  export SSRS_LOG_DIR="$STRUCT_DIR"
  export SSRS_LOSS_MODE="SEG+BDY+OBJ"
  export SSRS_USE_STRUCTURE_LOSS="1"
  python train.py 2>&1 | tee "$STRUCT_DIR/train.log"
)

echo "[Week2] Building comparison markdown..."
python - <<'PY'
import csv
import os

week2_root = os.environ["WEEK2_ROOT"] if "WEEK2_ROOT" in os.environ else None
if week2_root is None:
    raise RuntimeError("WEEK2_ROOT env var is missing")

base_csv = os.path.join(week2_root, "baseline_seg", f"MFNet_{os.environ.get('SSRS_DATASET','Vaihingen')}_seed{os.environ.get('SSRS_SEED','42')}.csv")
struct_csv = os.path.join(week2_root, "struct_seg_bdy_obj", f"MFNet_{os.environ.get('SSRS_DATASET','Vaihingen')}_seed{os.environ.get('SSRS_SEED','42')}.csv")
out_md = os.path.join(week2_root, "week2_same_domain_summary.md")

def read_best(csv_path):
    if not os.path.isfile(csv_path):
        return None
    best = None
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                metric = float(row.get("val_metric", "nan"))
            except Exception:
                continue
            if best is None or metric > best["metric"]:
                best = {
                    "metric": metric,
                    "epoch": row.get("epoch", "NA"),
                    "loss_mode": row.get("loss_mode", "NA"),
                    "train_loss": row.get("train_loss", "NA"),
                    "mean_f1": row.get("mean_f1", "NA"),
                    "kappa": row.get("kappa", "NA"),
                    "mean_miou": row.get("mean_miou", "NA"),
                }
    return best

base = read_best(base_csv)
struct = read_best(struct_csv)

def fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)

lines = []
lines.append("# Week-2 Same-Domain Summary")
lines.append("")
lines.append("| Run | Loss Mode | Best Val Metric | Best Epoch | Train Loss | mean F1 | Kappa | mean MIoU |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
if base:
    lines.append(f"| Baseline | {fmt(base['loss_mode'])} | {fmt(base['metric'])} | {fmt(base['epoch'],0)} | {fmt(base['train_loss'])} | {fmt(base['mean_f1'])} | {fmt(base['kappa'])} | {fmt(base['mean_miou'])} |")
else:
    lines.append("| Baseline | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
if struct:
    lines.append(f"| Structured | {fmt(struct['loss_mode'])} | {fmt(struct['metric'])} | {fmt(struct['epoch'],0)} | {fmt(struct['train_loss'])} | {fmt(struct['mean_f1'])} | {fmt(struct['kappa'])} | {fmt(struct['mean_miou'])} |")
else:
    lines.append("| Structured | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

if base and struct:
    delta = struct["metric"] - base["metric"]
    lines.append("")
    lines.append(f"- Delta(Structured - Baseline): {delta:.4f}")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[Week2] Summary written: {out_md}")
PY

echo "[Week2] Done."
echo "[Week2] Baseline log dir:   $BASELINE_DIR"
echo "[Week2] Structured log dir: $STRUCT_DIR"
echo "[Week2] Summary markdown:   $WEEK2_ROOT/week2_same_domain_summary.md"
