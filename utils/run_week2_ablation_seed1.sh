#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

# -----------------------------
# 基础参数（可在外部 export 覆盖）
# -----------------------------
export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_DATASET="${SSRS_DATASET:-Vaihingen}"
export SSRS_SEED="${SSRS_SEED:-42}"                     # 单种子默认 42
export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE="${SSRS_BATCH_SIZE:-16}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"

export SSRS_PERF_MODE="${SSRS_PERF_MODE:-1}"
export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-16}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-1}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-6}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"
export SSRS_MFNET_MICRO_BS="${SSRS_MFNET_MICRO_BS:-8}"

# 快筛默认（20/400）；正式可改为 50/1000
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-400}"
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_SAVE_EPOCH="${SSRS_SAVE_EPOCH:-1}"
export SSRS_SAVE_BEST="${SSRS_SAVE_BEST:-1}"
export SSRS_SAVE_LAST="${SSRS_SAVE_LAST:-1}"
export SSRS_SAVE_INTERVAL="${SSRS_SAVE_INTERVAL:-0}"
export SSRS_EVAL_NUM_TILES="${SSRS_EVAL_NUM_TILES:-0}"

# 结构损失参数（默认沿用你当前设定）
export SSRS_LAMBDA_BDY="${SSRS_LAMBDA_BDY:-0.1}"
export SSRS_LAMBDA_OBJ="${SSRS_LAMBDA_OBJ:-1.0}"
export SSRS_STRUCTURE_WARMUP_EPOCHS="${SSRS_STRUCTURE_WARMUP_EPOCHS:-10}"
export SSRS_STRUCTURE_CONF_THRESH="${SSRS_STRUCTURE_CONF_THRESH:-0.6}"
export SSRS_PRIOR_QUALITY_CHECK="${SSRS_PRIOR_QUALITY_CHECK:-1}"
export SSRS_PRIOR_QUALITY_STRICT="${SSRS_PRIOR_QUALITY_STRICT:-0}"
export SSRS_PRIOR_CHECK_SAMPLES="${SSRS_PRIOR_CHECK_SAMPLES:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

# 线程数兜底：避免外部环境传入 0/负数/非数字导致 libgomp 报错。
if [[ ! "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Week2-Ablation][WARN] Invalid OMP_NUM_THREADS='$OMP_NUM_THREADS', fallback to 16"
    export OMP_NUM_THREADS=16
fi
if [[ ! "$MKL_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Week2-Ablation][WARN] Invalid MKL_NUM_THREADS='$MKL_NUM_THREADS', fallback to 16"
    export MKL_NUM_THREADS=16
fi

TS="$(date +%Y%m%d_%H%M%S)"
WEEK2_ROOT="${SSRS_WEEK2_OUT_DIR:-$ROOT_DIR/runs/week2_ablation_seed1/$TS}"
BASE_DIR="$WEEK2_ROOT/seg"
BDY_DIR="$WEEK2_ROOT/seg_bdy"
OBJ_DIR="$WEEK2_ROOT/seg_obj"
BOTH_DIR="$WEEK2_ROOT/seg_bdy_obj"
mkdir -p "$BASE_DIR" "$BDY_DIR" "$OBJ_DIR" "$BOTH_DIR"
export WEEK2_ROOT

echo "[Week2-Ablation] Root:        $ROOT_DIR"
echo "[Week2-Ablation] Data root:   $SSRS_DATA_ROOT"
echo "[Week2-Ablation] Dataset:     $SSRS_DATASET"
echo "[Week2-Ablation] Seed:        $SSRS_SEED"
echo "[Week2-Ablation] EPOCHS/STEPS:$SSRS_EPOCHS/$SSRS_EPOCH_STEPS"
echo "[Week2-Ablation] Lambda:      bdy=$SSRS_LAMBDA_BDY obj=$SSRS_LAMBDA_OBJ"
echo "[Week2-Ablation] Warmup/conf: $SSRS_STRUCTURE_WARMUP_EPOCHS / $SSRS_STRUCTURE_CONF_THRESH"
echo "[Week2-Ablation] Out:         $WEEK2_ROOT"

if [[ ! -d "$SSRS_DATA_ROOT" ]]; then
  echo "[Week2-Ablation][FAIL] SSRS_DATA_ROOT not found: $SSRS_DATA_ROOT"
  exit 1
fi
if [[ ! -f "$MFNET_DIR/train.py" ]]; then
  echo "[Week2-Ablation][FAIL] train.py not found: $MFNET_DIR/train.py"
  exit 1
fi

run_one() {
  local name="$1"
  local out_dir="$2"
  local loss_mode="$3"
  local use_structure="$4"

  echo "[Week2-Ablation] Run => $name ($loss_mode)"
  (
    cd "$MFNET_DIR"
    export SSRS_LOG_DIR="$out_dir"
    export SSRS_LOSS_MODE="$loss_mode"
    export SSRS_USE_STRUCTURE_LOSS="$use_structure"
    python train.py 2>&1 | tee "$out_dir/train.log"
  )
}

# 1) Baseline
run_one "Baseline" "$BASE_DIR" "SEG" "0"
# 2) +Boundary
run_one "+Boundary" "$BDY_DIR" "SEG+BDY" "1"
# 3) +Object
run_one "+Object" "$OBJ_DIR" "SEG+OBJ" "1"
# 4) +Boundary+Object
run_one "+Boundary+Object" "$BOTH_DIR" "SEG+BDY+OBJ" "1"

# 汇总 markdown
python - <<'PY'
import csv
import os

week2_root = os.environ["WEEK2_ROOT"]
dataset = os.environ.get("SSRS_DATASET", "Vaihingen")
seed = os.environ.get("SSRS_SEED", "42")
out_md = os.path.join(week2_root, "week2_ablation_seed1_summary.md")

runs = [
    ("Baseline", "seg"),
    ("+Boundary", "seg_bdy"),
    ("+Object", "seg_obj"),
    ("+Boundary+Object", "seg_bdy_obj"),
]


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
                    "mean_f1": row.get("mean_f1", "NA"),
                    "kappa": row.get("kappa", "NA"),
                    "mean_miou": row.get("mean_miou", "NA"),
                    "train_loss": row.get("train_loss", "NA"),
                    "train_loss_ce": row.get("train_loss_ce", "NA"),
                    "train_loss_boundary": row.get("train_loss_boundary", "NA"),
                    "train_loss_object": row.get("train_loss_object", "NA"),
                    "warmup_scale": row.get("warmup_scale", "NA"),
                    "conf_threshold": row.get("conf_threshold", "NA"),
                    "conf_kept_ratio": row.get("conf_kept_ratio", "NA"),
                }
    return best


def fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)

stats = []
for run_name, subdir in runs:
    csv_path = os.path.join(week2_root, subdir, f"MFNet_{dataset}_seed{seed}.csv")
    stats.append((run_name, read_best(csv_path)))

base_metric = stats[0][1]["metric"] if stats[0][1] else None

lines = []
lines.append("# Week-2 Same-Domain Ablation (Single Seed)")
lines.append("")
lines.append(f"- dataset: {dataset}")
lines.append(f"- seed: {seed}")
lines.append("")
lines.append("| Run | Loss Mode | Best Val Metric | Delta vs Baseline | Best Epoch | mean F1 | Kappa | mean MIoU | CE | BDY | OBJ | warmup_scale | conf_kept_ratio |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for run_name, st in stats:
    if not st:
        lines.append(f"| {run_name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        continue

    delta = "N/A"
    if base_metric is not None:
        delta = f"{(st['metric'] - base_metric):.4f}"

    lines.append(
        f"| {run_name} | {fmt(st['loss_mode'])} | {fmt(st['metric'])} | {delta} | {fmt(st['epoch'],0)} | "
        f"{fmt(st['mean_f1'])} | {fmt(st['kappa'])} | {fmt(st['mean_miou'])} | "
        f"{fmt(st['train_loss_ce'])} | {fmt(st['train_loss_boundary'])} | {fmt(st['train_loss_object'])} | "
        f"{fmt(st['warmup_scale'])} | {fmt(st['conf_kept_ratio'])} |"
    )

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[Week2-Ablation] Summary written: {out_md}")
PY

echo "[Week2-Ablation] Done"
echo "[Week2-Ablation] Output root: $WEEK2_ROOT"
echo "[Week2-Ablation] Summary:     $WEEK2_ROOT/week2_ablation_seed1_summary.md"
