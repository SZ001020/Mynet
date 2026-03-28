#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-$ROOT_DIR/autodl-tmp/dataset}"
export SSRS_DATASET="${SSRS_DATASET:-Vaihingen}"
export SSRS_SEED="${SSRS_SEED:-42}"
export SSRS_LOG_DIR="${SSRS_LOG_DIR:-$ROOT_DIR/runs/week1_baseline}"
export SSRS_BATCH_SIZE="${SSRS_BATCH_SIZE:-10}"
export SSRS_WINDOW_SIZE="${SSRS_WINDOW_SIZE:-256}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-1000}"
export SSRS_EPOCHS="${SSRS_EPOCHS:-50}"
export SSRS_SAVE_EPOCH="${SSRS_SAVE_EPOCH:-1}"
export SSRS_SAVE_BEST="${SSRS_SAVE_BEST:-1}"
export SSRS_SAVE_LAST="${SSRS_SAVE_LAST:-1}"
export SSRS_SAVE_INTERVAL="${SSRS_SAVE_INTERVAL:-0}"
export SSRS_EVAL_NUM_TILES="${SSRS_EVAL_NUM_TILES:-0}"
export SSRS_PERF_MODE="${SSRS_PERF_MODE:-1}"
export SSRS_NUM_WORKERS="${SSRS_NUM_WORKERS:-16}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-1}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-4}"

# Avoid invalid OpenMP/MKL thread settings (0 is invalid and slows runtime initialization).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
if [[ "$OMP_NUM_THREADS" -le 0 ]]; then
	export OMP_NUM_THREADS=16
fi
if [[ "$MKL_NUM_THREADS" -le 0 ]]; then
	export MKL_NUM_THREADS=16
fi

mkdir -p "$SSRS_LOG_DIR"

echo "[Week1] Data root: $SSRS_DATA_ROOT"
echo "[Week1] Dataset:   $SSRS_DATASET"
echo "[Week1] Seed:      $SSRS_SEED"
echo "[Week1] Log dir:   $SSRS_LOG_DIR"
echo "[Week1] Epoch steps: $SSRS_EPOCH_STEPS"
echo "[Week1] Epochs:      $SSRS_EPOCHS"
echo "[Week1] Save every:  $SSRS_SAVE_EPOCH"
echo "[Week1] Save best:   $SSRS_SAVE_BEST"
echo "[Week1] Save last:   $SSRS_SAVE_LAST"
echo "[Week1] Save interval snapshots: $SSRS_SAVE_INTERVAL (0 disables)"
echo "[Week1] Eval tiles:  $SSRS_EVAL_NUM_TILES (0 means all)"
echo "[Week1] Perf mode:   $SSRS_PERF_MODE"
echo "[Week1] Workers:     $SSRS_NUM_WORKERS"
echo "[Week1] OMP/MKL:     $OMP_NUM_THREADS/$MKL_NUM_THREADS"

echo "[Week1] Running MFNet..."
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"
export SSRS_MFNET_MICRO_BS="${SSRS_MFNET_MICRO_BS:-2}"
pushd "$ROOT_DIR/MFNet" >/dev/null
python train.py
popd >/dev/null

echo "[Week1] Running FTransUNet..."
pushd "$ROOT_DIR/FTransUNet" >/dev/null
python train.py
popd >/dev/null

echo "[Week1] Running ASMFNet..."
pushd "$ROOT_DIR/ASMFNet" >/dev/null
python train.py
popd >/dev/null

echo "[Week1] Building summary..."
python "$ROOT_DIR/utils/summarize_week1_baseline.py"

echo "[Week1] Done. Summary at: $SSRS_LOG_DIR/week1_baseline_summary.md"
