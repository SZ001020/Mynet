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

# Recommend short-run first for grid search.
export SSRS_EPOCH_STEPS="${SSRS_EPOCH_STEPS:-400}"
export SSRS_EPOCHS="${SSRS_EPOCHS:-20}"
export SSRS_SAVE_EPOCH="${SSRS_SAVE_EPOCH:-1}"
export SSRS_SAVE_BEST="${SSRS_SAVE_BEST:-1}"
export SSRS_SAVE_LAST="${SSRS_SAVE_LAST:-1}"
export SSRS_SAVE_INTERVAL="${SSRS_SAVE_INTERVAL:-0}"
export SSRS_EVAL_NUM_TILES="${SSRS_EVAL_NUM_TILES:-0}"

export SSRS_USE_STRUCTURE_LOSS=1
export SSRS_REQUIRE_STRUCTURE_PRIORS="${SSRS_REQUIRE_STRUCTURE_PRIORS:-1}"
export SSRS_LOSS_MODE="SEG+BDY+OBJ"

# Default grid from current diagnosis.
B_LAMBDAS="${SSRS_GRID_LAMBDA_BDY:-0.01,0.03,0.05}"
O_LAMBDAS="${SSRS_GRID_LAMBDA_OBJ:-0.1,0.3,0.5}"

TS="$(date +%Y%m%d_%H%M%S)"
GRID_ROOT="${SSRS_WEEK2_GRID_OUT_DIR:-$ROOT_DIR/runs/week2_lambda_grid/$TS}"
mkdir -p "$GRID_ROOT"

RESULT_CSV="$GRID_ROOT/grid_results.csv"
echo "lambda_bdy,lambda_obj,best_epoch,best_val_metric,mean_f1,kappa,mean_miou,csv_path" > "$RESULT_CSV"

echo "[Week2-Grid] Root: $GRID_ROOT"
echo "[Week2-Grid] Dataset: $SSRS_DATASET, seed=$SSRS_SEED"
echo "[Week2-Grid] Epochs/steps: $SSRS_EPOCHS/$SSRS_EPOCH_STEPS"

IFS=',' read -r -a B_ARR <<< "$B_LAMBDAS"
IFS=',' read -r -a O_ARR <<< "$O_LAMBDAS"

for lb in "${B_ARR[@]}"; do
  for lo in "${O_ARR[@]}"; do
    tag="bdy_${lb}_obj_${lo}"
    out_dir="$GRID_ROOT/$tag"
    mkdir -p "$out_dir"

    echo "[Week2-Grid] Running $tag ..."
    (
      cd "$MFNET_DIR"
      export SSRS_LOG_DIR="$out_dir"
      export SSRS_LAMBDA_BDY="$lb"
      export SSRS_LAMBDA_OBJ="$lo"
      python train.py 2>&1 | tee "$out_dir/train.log"
    )

    csv_path="$out_dir/MFNet_${SSRS_DATASET}_seed${SSRS_SEED}.csv"
    if [[ ! -f "$csv_path" ]]; then
      echo "[Week2-Grid][WARN] Missing csv: $csv_path"
      continue
    fi

    python - <<PY >> "$RESULT_CSV"
import csv
csv_path = r"$csv_path"
lb = "$lb"
lo = "$lo"
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
                "mean_f1": row.get("mean_f1", "NA"),
                "kappa": row.get("kappa", "NA"),
                "mean_miou": row.get("mean_miou", "NA"),
            }
if best is None:
    print(f"{lb},{lo},NA,NA,NA,NA,NA,{csv_path}")
else:
    print(f"{lb},{lo},{best['epoch']},{best['metric']},{best['mean_f1']},{best['kappa']},{best['mean_miou']},{csv_path}")
PY
  done
done

SUMMARY_MD="$GRID_ROOT/grid_summary.md"
{
  echo "# Week-2 Lambda Grid Summary"
  echo
  echo "- Dataset: $SSRS_DATASET"
  echo "- Seed: $SSRS_SEED"
  echo "- Epochs/steps: $SSRS_EPOCHS/$SSRS_EPOCH_STEPS"
  echo
  echo "| lambda_bdy | lambda_obj | best_epoch | best_val_metric | mean_f1 | kappa | mean_miou |"
  echo "|---:|---:|---:|---:|---:|---:|---:|"
  tail -n +2 "$RESULT_CSV" | awk -F',' '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7}'
  echo
  best_line="$(tail -n +2 "$RESULT_CSV" | awk -F',' '$4!="NA" {print}' | sort -t',' -k4,4nr | head -n 1)"
  if [[ -n "$best_line" ]]; then
    bdy="$(echo "$best_line" | awk -F',' '{print $1}')"
    obj="$(echo "$best_line" | awk -F',' '{print $2}')"
    metric="$(echo "$best_line" | awk -F',' '{print $4}')"
    echo "## Recommended Next Setting"
    echo
    echo "- lambda_bdy: $bdy"
    echo "- lambda_obj: $obj"
    echo "- best_val_metric: $metric"
  else
    echo "## Recommended Next Setting"
    echo
    echo "- No valid run found."
  fi
} > "$SUMMARY_MD"

echo "[Week2-Grid] Done."
echo "[Week2-Grid] CSV: $RESULT_CSV"
echo "[Week2-Grid] Summary: $SUMMARY_MD"
