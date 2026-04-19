#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:-$ROOT_DIR/autodl-tmp/runs/week5/20260415_161039}"
NUM_TILES="${SSRS_DAY4_NUM_TILES:-3}"
BATCH_SIZE="${SSRS_DAY4_BATCH_SIZE:-8}"
STRIDE="${SSRS_DAY4_STRIDE:-128}"
NUM_ZOOMS="${SSRS_DAY4_NUM_ZOOMS:-3}"
ZOOM_SIZE="${SSRS_DAY4_ZOOM_SIZE:-384}"
MAX_PANEL_WIDTH="${SSRS_DAY4_MAX_PANEL_WIDTH:-3200}"
PNG_LEVEL="${SSRS_DAY4_PNG_COMPRESS_LEVEL:-9}"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "[Week5-Day4][ERR] run root not found: $RUN_ROOT"
  exit 1
fi

echo "[Week5-Day4] run root: $RUN_ROOT"
echo "[Week5-Day4] num_tiles=$NUM_TILES stride=$STRIDE batch_size=$BATCH_SIZE"
echo "[Week5-Day4] num_zooms=$NUM_ZOOMS zoom_size=$ZOOM_SIZE max_panel_width=$MAX_PANEL_WIDTH png_level=$PNG_LEVEL"

python "$ROOT_DIR/utils/generate_week5_day4_visuals.py" \
  --run-root "$RUN_ROOT" \
  --num-tiles "$NUM_TILES" \
  --batch-size "$BATCH_SIZE" \
  --stride "$STRIDE" \
  --num-zooms "$NUM_ZOOMS" \
  --zoom-size "$ZOOM_SIZE" \
  --max-panel-width "$MAX_PANEL_WIDTH" \
  --png-compress-level "$PNG_LEVEL"

echo "[Week5-Day4] done"
