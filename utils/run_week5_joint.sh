#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_SOURCE_DATASET="${SSRS_SOURCE_DATASET:-Vaihingen}"
export SSRS_TARGET_DATASET="${SSRS_TARGET_DATASET:-Potsdam}"
export SSRS_SEED="${SSRS_SEED:-42}"

# Week5: keep the same default hyper-parameters as week4/day2 script.
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-1000}"
export SSRS_EVAL_EVERY="${SSRS_EVAL_EVERY:-5}"
export SSRS_EVAL_MAX_TILES="${SSRS_EVAL_MAX_TILES:-0}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"

export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE_UDA="${SSRS_BATCH_SIZE_UDA:-6}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"

# Week5 main UDA config (single config for 5-group main table)
export SSRS_WEEK5_MAIN_LAMBDA_ADV="${SSRS_WEEK5_MAIN_LAMBDA_ADV:-0.0010}"
export SSRS_WEEK5_MAIN_GRL_LAMBDA="${SSRS_WEEK5_MAIN_GRL_LAMBDA:-1.0}"
export SSRS_WEEK5_MAIN_ADV_LR="${SSRS_WEEK5_MAIN_ADV_LR:-0.0010}"
export SSRS_WEEK5_MAIN_LAMBDA_BDY="${SSRS_WEEK5_MAIN_LAMBDA_BDY:-0.1}"
export SSRS_WEEK5_MAIN_LAMBDA_OBJ="${SSRS_WEEK5_MAIN_LAMBDA_OBJ:-1.0}"
export SSRS_WEEK5_STRUCTURE_WARMUP_EPOCHS="${SSRS_WEEK5_STRUCTURE_WARMUP_EPOCHS:-10}"
export SSRS_WEEK5_STRUCTURE_CONF_THRESH="${SSRS_WEEK5_STRUCTURE_CONF_THRESH:-0.6}"
export SSRS_REQUIRE_STRUCTURE_PRIORS="${SSRS_REQUIRE_STRUCTURE_PRIORS:-0}"

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
# Keep the same default save path style as the week5 script.
OUT_ROOT="${SSRS_WEEK5_OUT_DIR:-/root/autodl-tmp/runs/week5/$TS}"
mkdir -p "$OUT_ROOT"

printf '[Week5] source=%s target=%s seed=%s\n' "$SSRS_SOURCE_DATASET" "$SSRS_TARGET_DATASET" "$SSRS_SEED"
printf '[Week5] out=%s\n' "$OUT_ROOT"
printf '[Week5] epochs/steps=%s/%s\n' "$SSRS_EPOCHS" "$SSRS_EPOCH_STEPS"
printf '[Week5] main_uda=lambda_adv=%s grl=%s adv_lr=%s\n' \
  "$SSRS_WEEK5_MAIN_LAMBDA_ADV" "$SSRS_WEEK5_MAIN_GRL_LAMBDA" "$SSRS_WEEK5_MAIN_ADV_LR"
printf '[Week5] structure=lambda_bdy=%s lambda_obj=%s warmup=%s conf=%.2f require_priors=%s\n' \
    "$SSRS_WEEK5_MAIN_LAMBDA_BDY" "$SSRS_WEEK5_MAIN_LAMBDA_OBJ" "$SSRS_WEEK5_STRUCTURE_WARMUP_EPOCHS" "$SSRS_WEEK5_STRUCTURE_CONF_THRESH" "$SSRS_REQUIRE_STRUCTURE_PRIORS"

# Week5 planned 5 groups:
# source_only / uda / uda_bdy / uda_obj / uda_bdy_obj
EXP_GROUPS=(
    "source_only|source-only|SEG|0|source_only_seed${SSRS_SEED}"
    "uda|uda-high|SEG|0|uda_seed${SSRS_SEED}"
    "uda_bdy|uda-high|SEG+BDY|1|uda_bdy_seed${SSRS_SEED}"
    "uda_obj|uda-high|SEG+OBJ|1|uda_obj_seed${SSRS_SEED}"
    "uda_bdy_obj|uda-high|SEG+BDY+OBJ|1|uda_bdy_obj_seed${SSRS_SEED}"
)

for item in "${EXP_GROUPS[@]}"; do
    IFS='|' read -r GROUP MODE_NAME LOSS_MODE USE_STRUCTURE run_id <<< "$item"
    if [[ -z "${GROUP:-}" || -z "${MODE_NAME:-}" || -z "${LOSS_MODE:-}" || -z "${USE_STRUCTURE:-}" || -z "${run_id:-}" ]]; then
        echo "[Week5][ERROR] invalid GROUPS item: $item"
        exit 2
    fi
  run_dir="$OUT_ROOT/$run_id"
  mkdir -p "$run_dir"

    printf '[Week5][RUN] %s (%s, %s)\n' "$run_id" "$MODE_NAME" "$LOSS_MODE"
  (
    cd "$MFNET_DIR"
    export SSRS_LOG_DIR="$run_dir"
    export SSRS_WEEK3_MODE="$MODE_NAME"
        export SSRS_LOSS_MODE="$LOSS_MODE"
        export SSRS_USE_STRUCTURE_LOSS="$USE_STRUCTURE"
    export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_UDA"
    export SSRS_LAMBDA_ADV="$SSRS_WEEK5_MAIN_LAMBDA_ADV"
    export SSRS_GRL_LAMBDA="$SSRS_WEEK5_MAIN_GRL_LAMBDA"
    export SSRS_ADV_LR="$SSRS_WEEK5_MAIN_ADV_LR"
        export SSRS_LAMBDA_BDY="$SSRS_WEEK5_MAIN_LAMBDA_BDY"
        export SSRS_LAMBDA_OBJ="$SSRS_WEEK5_MAIN_LAMBDA_OBJ"
        export SSRS_STRUCTURE_WARMUP_EPOCHS="$SSRS_WEEK5_STRUCTURE_WARMUP_EPOCHS"
        export SSRS_STRUCTURE_CONF_THRESH="$SSRS_WEEK5_STRUCTURE_CONF_THRESH"
        export SSRS_REQUIRE_STRUCTURE_PRIORS="$SSRS_REQUIRE_STRUCTURE_PRIORS"
    python train_uda_struct_v1.py 2>&1 | tee "$run_dir/train.log"
  )
done

export OUT_ROOT
export SSRS_SEED
python - << 'PY'
import csv
import math
import os
from glob import glob

out_root = os.environ['OUT_ROOT']
seed = os.environ['SSRS_SEED']

planned = [
    ('source_only', f'source_only_seed{seed}'),
    ('uda', f'uda_seed{seed}'),
    ('uda_bdy', f'uda_bdy_seed{seed}'),
    ('uda_obj', f'uda_obj_seed{seed}'),
    ('uda_bdy_obj', f'uda_bdy_obj_seed{seed}'),
]

rows = []
rank_pool = []

for group_name, run_id in planned:
    run_dir = os.path.join(out_root, run_id)
    csv_list = glob(os.path.join(run_dir, 'MFNet_week3_*seed*.csv'))

    if not csv_list:
        rows.append((group_name, run_id, 'MISSING', float('nan'), 'NA', 'no csv found'))
        continue

    best_m = None
    best_e = 'NA'
    for p in csv_list:
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
        rows.append((group_name, run_id, 'DONE', float('nan'), 'NA', 'eval missing in csv'))
    else:
        rows.append((group_name, run_id, 'DONE', best_m, best_e, ''))
        rank_pool.append((run_id, best_m, best_e))

rank_pool.sort(key=lambda x: x[1], reverse=True)

sum_md = os.path.join(out_root, 'week5_joint_summary.md')
with open(sum_md, 'w', encoding='utf-8') as f:
    f.write('# Week5 Joint Summary\n\n')
    f.write('## Plan-aligned group status\n\n')
    f.write('| group | run_id | status | best_mIoU | epoch | note |\n')
    f.write('|---|---|---|---:|---:|---|\n')
    for group_name, run_id, status, m, e, note in rows:
        m_text = f'{m:.4f}' if not math.isnan(m) else 'NA'
        f.write(f'| {group_name} | {run_id} | {status} | {m_text} | {e} | {note} |\n')

    f.write('\n## Ranking among completed groups\n\n')
    f.write('| rank | run_id | best_mIoU | epoch |\n')
    f.write('|---:|---|---:|---:|\n')
    for i, (rid, m, e) in enumerate(rank_pool, 1):
        f.write(f'| {i} | {rid} | {m:.4f} | {e} |\n')

    f.write('\n## Notes\n\n')
    f.write('- This script is aligned to the Week5 five-group plan in directory structure and summary format.\n')
    f.write('- If structure-prior files are incomplete, set SSRS_REQUIRE_STRUCTURE_PRIORS=0 to fallback to zero priors.\n')
print('[Week5] summary:', sum_md)
PY

printf '[Week5] done: %s\n' "$OUT_ROOT"
