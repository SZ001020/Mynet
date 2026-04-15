#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_SOURCE_DATASET="${SSRS_SOURCE_DATASET:-Vaihingen}"
export SSRS_TARGET_DATASET="${SSRS_TARGET_DATASET:-Potsdam}"
export SSRS_SEED="${SSRS_SEED:-42}"

# Day2: refine runs with the same default hyper-parameters as Day1.
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-400}"
export SSRS_EVAL_EVERY="${SSRS_EVAL_EVERY:-5}"
export SSRS_EVAL_MAX_TILES="${SSRS_EVAL_MAX_TILES:-0}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"

export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
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
OUT_ROOT="${SSRS_WEEK4_DAY2_OUT_DIR:-/root/autodl-tmp/runs/week4_stabilization/day2_refine/$TS}"
mkdir -p "$OUT_ROOT"

printf '[Day2] source=%s target=%s seed=%s\n' "$SSRS_SOURCE_DATASET" "$SSRS_TARGET_DATASET" "$SSRS_SEED"
printf '[Day2] out=%s\n' "$OUT_ROOT"
printf '[Day2] epochs/steps=%s/%s\n' "$SSRS_EPOCHS" "$SSRS_EPOCH_STEPS"

# Use top-3 configs from day1 grid summary.
COMBOS=(
    "c05 0.0030 1.0 0.0010"
    "c06 0.0020 1.5 0.0015"
    "c03 0.0010 1.0 0.0010"
)

for item in "${COMBOS[@]}"; do
  read -r TAG LAMBDA GRL ADVLR <<< "$item"
  run_id="${TAG}_l${LAMBDA}_g${GRL}_a${ADVLR}"
  run_dir="$OUT_ROOT/$run_id"
  mkdir -p "$run_dir"

  printf '[Day2][UDA] %s\n' "$run_id"
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

export OUT_ROOT
python - << 'PY'
import csv
import math
import os
from glob import glob

out_root = os.environ['OUT_ROOT']

rows = []
for p in glob(os.path.join(out_root, '*', 'MFNet_week3_*seed*.csv')):
    best_m = None
    best_e = 'NA'
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get('target_mean_miou', 'nan'))
            except Exception:
                continue
            if math.isnan(m):
                continue
            if best_m is None or m > best_m:
                best_m = m
                best_e = row.get('epoch', 'NA')
    if best_m is None:
        continue
    rid = os.path.basename(os.path.dirname(p))
    rows.append((rid, best_m, best_e))
rows.sort(key=lambda x: x[1], reverse=True)

sum_md = os.path.join(out_root, 'day2_refine_summary.md')
with open(sum_md, 'w', encoding='utf-8') as f:
    f.write('# Week4 Day2 Refine Summary\n\n')
    f.write('| rank | run_id | best_mIoU | epoch |\n')
    f.write('|---:|---|---:|---:|\n')
    for i, (rid, m, e) in enumerate(rows, 1):
        f.write(f'| {i} | {rid} | {m:.4f} | {e} |\n')
print('[Day2] summary:', sum_md)
PY

printf '[Day2] done: %s\n' "$OUT_ROOT"
