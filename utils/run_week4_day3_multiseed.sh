#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_SOURCE_DATASET="${SSRS_SOURCE_DATASET:-Vaihingen}"
export SSRS_TARGET_DATASET="${SSRS_TARGET_DATASET:-Potsdam}"

# Day3: multi-seed stability check for selected configs.
export SSRS_EPOCHS="${SSRS_EPOCHS:-30}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-300}"
export SSRS_EVAL_EVERY="${SSRS_EVAL_EVERY:-3}"
export SSRS_EVAL_MAX_TILES="${SSRS_EVAL_MAX_TILES:-0}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"

export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE_UDA="${SSRS_BATCH_SIZE_UDA:-6}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"

export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-2}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-0}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-2}"
export SSRS_DROP_LAST="${SSRS_DROP_LAST:-1}"
export SSRS_DATA_CACHE="${SSRS_DATA_CACHE:-1}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${SSRS_WEEK4_DAY3_OUT_DIR:-$ROOT_DIR/runs/week4_stabilization/day3_multiseed/$TS}"
mkdir -p "$OUT_ROOT"

printf '[Day3] source=%s target=%s\n' "$SSRS_SOURCE_DATASET" "$SSRS_TARGET_DATASET"
printf '[Day3] out=%s\n' "$OUT_ROOT"

# Replace with top-1/2 configs selected from day2 summary.
CONFIGS=(
  "bestA 0.0010 1.0 0.0010"
  "bestB 0.0020 1.0 0.0010"
)
SEEDS=(7 42 3407)

for cfg in "${CONFIGS[@]}"; do
  read -r CFG_TAG LAMBDA GRL ADVLR <<< "$cfg"
  for SEED in "${SEEDS[@]}"; do
    run_id="${CFG_TAG}_seed${SEED}"
    run_dir="$OUT_ROOT/$run_id"
    mkdir -p "$run_dir"

    printf '[Day3][UDA] %s\n' "$run_id"
    (
      cd "$MFNET_DIR"
      export SSRS_LOG_DIR="$run_dir"
      export SSRS_WEEK3_MODE="uda-high"
      export SSRS_SEED="$SEED"
      export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_UDA"
      export SSRS_LAMBDA_ADV="$LAMBDA"
      export SSRS_GRL_LAMBDA="$GRL"
      export SSRS_ADV_LR="$ADVLR"
      python train_uda_struct_v1.py 2>&1 | tee "$run_dir/train.log"
    )
  done
done

export OUT_ROOT
python - << 'PY'
import csv
import math
import os
from collections import defaultdict
from glob import glob

out_root = os.environ['OUT_ROOT']

raw = []
for p in glob(os.path.join(out_root, '*', 'MFNet_week3_*seed*.csv')):
    rid = os.path.basename(os.path.dirname(p))
    cfg = rid.split('_seed')[0]
    seed = rid.split('_seed')[-1]
    best_m = None
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
    if best_m is not None:
        raw.append((cfg, seed, best_m))

by_cfg = defaultdict(list)
for cfg, seed, m in raw:
    by_cfg[cfg].append(m)

sum_md = os.path.join(out_root, 'day3_multiseed_summary.md')
with open(sum_md, 'w', encoding='utf-8') as f:
    f.write('# Week4 Day3 Multi-seed Summary\n\n')
    f.write('## Raw best mIoU\n\n')
    f.write('| config | seed | best_mIoU |\n')
    f.write('|---|---:|---:|\n')
    for cfg, seed, m in sorted(raw):
        f.write(f'| {cfg} | {seed} | {m:.4f} |\n')

    f.write('\n## Mean and Std\n\n')
    f.write('| config | n | mean_mIoU | std_mIoU |\n')
    f.write('|---|---:|---:|---:|\n')
    for cfg, vals in sorted(by_cfg.items()):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = var ** 0.5
        f.write(f'| {cfg} | {n} | {mean:.4f} | {std:.4f} |\n')
print('[Day3] summary:', sum_md)
PY

printf '[Day3] done: %s\n' "$OUT_ROOT"
