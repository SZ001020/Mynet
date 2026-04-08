#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MFNET_DIR="$ROOT_DIR/MFNet"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_SOURCE_DATASET="${SSRS_SOURCE_DATASET:-Vaihingen}"
export SSRS_TARGET_DATASET="${SSRS_TARGET_DATASET:-Potsdam}"
export SSRS_SEED="${SSRS_SEED:-42}"

export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_BATCH_SIZE="${SSRS_BATCH_SIZE:-8}"
export SSRS_BATCH_SIZE_SRC="${SSRS_BATCH_SIZE_SRC:-$SSRS_BATCH_SIZE}"
export SSRS_BATCH_SIZE_UDA="${SSRS_BATCH_SIZE_UDA:-6}"
export SSRS_BASE_LR="${SSRS_BASE_LR:-0.01}"
export SSRS_ADV_LR="${SSRS_ADV_LR:-0.001}"
export SSRS_LAMBDA_ADV="${SSRS_LAMBDA_ADV:-0.001}"
export SSRS_GRL_LAMBDA="${SSRS_GRL_LAMBDA:-1.0}"

# Week3 prototype default: short-run first.
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-400}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"
export SSRS_EVAL_EVERY="${SSRS_EVAL_EVERY:-5}"
export SSRS_EVAL_MAX_TILES="${SSRS_EVAL_MAX_TILES:-0}"

export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-2}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-0}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-2}"
export SSRS_DROP_LAST="${SSRS_DROP_LAST:-1}"
export SSRS_DATA_CACHE="${SSRS_DATA_CACHE:-1}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
if [[ ! "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=16
fi
if [[ ! "$MKL_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=16
fi

TS="$(date +%Y%m%d_%H%M%S)"
WEEK3_ROOT="${SSRS_WEEK3_OUT_DIR:-$ROOT_DIR/runs/week3_weak_cross/$TS}"
SRC_ONLY_DIR="$WEEK3_ROOT/source_only"
UDA_DIR="$WEEK3_ROOT/uda_high"
mkdir -p "$SRC_ONLY_DIR" "$UDA_DIR"


echo "[Week3] source=${SSRS_SOURCE_DATASET}, target=${SSRS_TARGET_DATASET}, seed=${SSRS_SEED}"
echo "[Week3] epochs/steps=${SSRS_EPOCHS}/${SSRS_EPOCH_STEPS}"
echo "[Week3] batch source/uda=${SSRS_BATCH_SIZE_SRC}/${SSRS_BATCH_SIZE_UDA}"
echo "[Week3] eval every/max_tiles/stride=${SSRS_EVAL_EVERY}/${SSRS_EVAL_MAX_TILES}/${SSRS_EVAL_STRIDE}"
echo "[Week3] workers/pin/persist/prefetch/drop=${SSRS_NUM_WORKERS}/${SSRS_PIN_MEMORY}/${SSRS_PERSISTENT_WORKERS}/${SSRS_PREFETCH_FACTOR}/${SSRS_DROP_LAST}"
echo "[Week3] data_cache=${SSRS_DATA_CACHE} (0=off,1=on)"
echo "[Week3] out=$WEEK3_ROOT"

echo "[Week3] Run source-only"
(
  cd "$MFNET_DIR"
  export SSRS_LOG_DIR="$SRC_ONLY_DIR"
  export SSRS_WEEK3_MODE="source-only"
  export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_SRC"
  python train_uda_struct_v1.py 2>&1 | tee "$SRC_ONLY_DIR/train.log"
)

echo "[Week3] Run single-level UDA"
(
  cd "$MFNET_DIR"
  export SSRS_LOG_DIR="$UDA_DIR"
  export SSRS_WEEK3_MODE="uda-high"
  export SSRS_BATCH_SIZE="$SSRS_BATCH_SIZE_UDA"
  python train_uda_struct_v1.py 2>&1 | tee "$UDA_DIR/train.log"
)

echo "[Week3] Done"
echo "[Week3] source-only: $SRC_ONLY_DIR"
echo "[Week3] uda-high:    $UDA_DIR"
