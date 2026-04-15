#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_SOURCE_DATASET="${SSRS_SOURCE_DATASET:-Vaihingen}"
export SSRS_TARGET_DATASET="${SSRS_TARGET_DATASET:-Potsdam}"
export SSRS_SEED="${SSRS_SEED:-42}"

# Day1: short-run coarse grid.
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-400}"
export SSRS_EVAL_EVERY="${SSRS_EVAL_EVERY:-5}"
export SSRS_EVAL_MAX_TILES="${SSRS_EVAL_MAX_TILES:-0}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"

export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE_SRC="${SSRS_BATCH_SIZE_SRC:-12}"
export SSRS_BATCH_SIZE_UDA="${SSRS_BATCH_SIZE_UDA:-6}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"

export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-4}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-1}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-2}"
export SSRS_DROP_LAST="${SSRS_DROP_LAST:-1}"
export SSRS_DATA_CACHE="${SSRS_DATA_CACHE:-1}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
if [[ ! "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=12
fi
if [[ ! "$MKL_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=12
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${SSRS_WEEK4_DAY1_OUT_DIR:-$ROOT_DIR/runs/week4_stabilization/day1_grid/$TS}"
mkdir -p "$OUT_ROOT/source_only"
export OUT_ROOT

printf '[Day1] source=%s target=%s seed=%s\n' "$SSRS_SOURCE_DATASET" "$SSRS_TARGET_DATASET" "$SSRS_SEED"
printf '[Day1] out=%s\n' "$OUT_ROOT"
printf '[Day1] epochs/steps=%s/%s, eval_every=%s\n' "$SSRS_EPOCHS" "$SSRS_EPOCH_STEPS" "$SSRS_EVAL_EVERY"

# 1) source-only anchor
(
  cd "$MFNET_DIR"
  export SSRS_LOG_DIR="$OUT_ROOT/source_only"
  export SSRS_WEEK3_MODE="source-only"
  export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_SRC"
  python train_uda_struct_v1.py 2>&1 | tee "$OUT_ROOT/source_only/train.log"
)

# 2) coarse UDA grid (lambda_adv grl_lambda adv_lr)
COMBOS=(
  "0.0005 0.5 0.0005"
  "0.0010 0.5 0.0010"
  "0.0010 1.0 0.0010"
  "0.0020 1.0 0.0010"
  "0.0030 1.0 0.0010"
  "0.0020 1.5 0.0015"
)

idx=0
for item in "${COMBOS[@]}"; do
  idx=$((idx + 1))
  read -r LAMBDA GRL ADVLR <<< "$item"
  run_id=$(printf 'c%02d_l%s_g%s_a%s' "$idx" "$LAMBDA" "$GRL" "$ADVLR")
  run_dir="$OUT_ROOT/uda_grid/$run_id"
  mkdir -p "$run_dir"

  printf '[Day1][UDA] %s\n' "$run_id"
  (
    cd "$MFNET_DIR"
    export SSRS_LOG_DIR="$run_dir"
    export SSRS_WEEK3_MODE="uda-high"
    export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_UDA"
    export SSRS_LAMBDA_ADV="$LAMBDA"
    export SSRS_GRL_LAMBDA="$GRL"
    export SSRS_ADV_LR="$ADVLR"
    python train_uda_struct_v1.py 2>&1 | tee "$run_dir/train.log"
  )
done

# 3) summarize best metric for quick ranking
python - << 'PY'
import csv
import math
import os
from glob import glob

out_root = os.environ['OUT_ROOT'] if 'OUT_ROOT' in os.environ else ''
if not out_root:
    raise SystemExit('OUT_ROOT is empty')

def best_miou(csv_path):
    best = None
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get('target_mean_miou', 'nan'))
            except Exception:
                continue
            if math.isnan(m):
                continue
            if best is None or m > best[0]:
                best = (m, row.get('epoch', 'NA'))
    return best

rows = []
for p in glob(os.path.join(out_root, 'uda_grid', '*', 'MFNet_week3_*seed*.csv')):
    b = best_miou(p)
    if b is None:
        continue
    run_id = os.path.basename(os.path.dirname(p))
    rows.append((run_id, b[0], b[1]))
rows.sort(key=lambda x: x[1], reverse=True)

sum_md = os.path.join(out_root, 'day1_grid_summary.md')
with open(sum_md, 'w', encoding='utf-8') as f:
    f.write('# Week4 Day1 Grid Summary\n\n')
    f.write('| rank | run_id | best_mIoU | epoch |\n')
    f.write('|---:|---|---:|---:|\n')
    for i, (rid, m, e) in enumerate(rows, 1):
        f.write(f'| {i} | {rid} | {m:.4f} | {e} |\n')
print('[Day1] summary:', sum_md)
PY

printf '[Day1] done: %s\n' "$OUT_ROOT"
