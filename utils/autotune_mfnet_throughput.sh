#!/usr/bin/env bash
set -u

# MFNet throughput autotune (short-run benchmark)
# It runs tiny training jobs and reports both wall time and throughput.

ROOT_DIR="/root/SSRS"
MFNET_DIR="/root/SSRS/MFNet"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SSRS_TUNE_OUT_DIR:-$ROOT_DIR/runs/autotune_mfnet_$TS}"
LOG_DIR="$OUT_DIR/job_logs"
mkdir -p "$LOG_DIR"

# Comma-separated candidate lists
WORKERS_LIST="${SSRS_TUNE_WORKERS_LIST:-12,16,20}"
MICRO_BS_LIST="${SSRS_TUNE_MICRO_BS_LIST:-2,3,4}"
BATCH_SIZE_LIST="${SSRS_TUNE_BATCH_SIZE_LIST:-8,10}"

# Short benchmark controls
TUNE_EPOCHS="${SSRS_TUNE_EPOCHS:-1}"
TUNE_EPOCH_STEPS="${SSRS_TUNE_EPOCH_STEPS:-120}"

# Shared env defaults
export SSRS_DATA_ROOT="${SSRS_DATA_ROOT:-/root/SSRS/autodl-tmp/dataset}"
export SSRS_DATASET="${SSRS_DATASET:-Vaihingen}"
export SSRS_SEED="${SSRS_SEED:-42}"
export SSRS_EVAL_STRIDE="${SSRS_EVAL_STRIDE:-32}"
export SSRS_PERF_MODE="${SSRS_PERF_MODE:-1}"
export SSRS_PIN_MEMORY="${SSRS_PIN_MEMORY:-1}"
export SSRS_PERSISTENT_WORKERS="${SSRS_PERSISTENT_WORKERS:-1}"
export SSRS_PREFETCH_FACTOR="${SSRS_PREFETCH_FACTOR:-4}"
export SSRS_MFNET_USE_AMP="${SSRS_MFNET_USE_AMP:-1}"

# Fix invalid thread envs (0 is invalid)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
if [[ "$OMP_NUM_THREADS" -le 0 ]]; then
  export OMP_NUM_THREADS=16
fi
if [[ "$MKL_NUM_THREADS" -le 0 ]]; then
  export MKL_NUM_THREADS=16
fi

RESULT_CSV="$OUT_DIR/results.csv"
{
  echo "workers,micro_bs,batch_size,epochs,epoch_steps,samples,elapsed_sec,throughput_samples_per_sec,status,log_file"
} > "$RESULT_CSV"

IFS=',' read -r -a WORKERS_ARR <<< "$WORKERS_LIST"
IFS=',' read -r -a MICRO_ARR <<< "$MICRO_BS_LIST"
IFS=',' read -r -a BATCH_ARR <<< "$BATCH_SIZE_LIST"

printf "[Autotune] Output dir: %s\n" "$OUT_DIR"
printf "[Autotune] Dataset: %s\n" "$SSRS_DATASET"
printf "[Autotune] Candidates workers=%s micro_bs=%s batch_size=%s\n" "$WORKERS_LIST" "$MICRO_BS_LIST" "$BATCH_SIZE_LIST"
printf "[Autotune] Short run: epochs=%s epoch_steps=%s\n" "$TUNE_EPOCHS" "$TUNE_EPOCH_STEPS"

for w in "${WORKERS_ARR[@]}"; do
  for m in "${MICRO_ARR[@]}"; do
    for b in "${BATCH_ARR[@]}"; do
      tag="w${w}_m${m}_b${b}"
      log_file="$LOG_DIR/$tag.log"

      printf "\n[Autotune] Running %s ...\n" "$tag"
      start_ts="$(date +%s)"

      set +e
      (
        cd "$MFNET_DIR" || exit 1
        export SSRS_NUM_WORKERS="$w"
        export SSRS_MFNET_MICRO_BS="$m"
        export SSRS_BATCH_SIZE="$b"
        export SSRS_EPOCHS="$TUNE_EPOCHS"
        export SSRS_EPOCH_STEPS="$TUNE_EPOCH_STEPS"
        export SSRS_SAVE_EPOCH=9999
        export SSRS_EVAL_NUM_TILES=0
        export SSRS_LOG_DIR="$OUT_DIR/train_logs/$tag"
        mkdir -p "$SSRS_LOG_DIR"
        python train.py
      ) >"$log_file" 2>&1
      status=$?
      set -e

      end_ts="$(date +%s)"
      elapsed=$((end_ts - start_ts))
      samples=$((TUNE_EPOCHS * TUNE_EPOCH_STEPS * b))
      throughput="$(awk -v s="$samples" -v t="$elapsed" 'BEGIN{ if (t<=0) printf "0.0000"; else printf "%.4f", s/t }')"

      if [[ "$status" -eq 0 ]]; then
        run_status="ok"
      else
        run_status="fail($status)"
      fi

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$w" "$m" "$b" "$TUNE_EPOCHS" "$TUNE_EPOCH_STEPS" "$samples" "$elapsed" "$throughput" "$run_status" "$log_file" \
        >> "$RESULT_CSV"

      printf "[Autotune] %s -> %s sec, %.4f samples/s, %s\n" "$tag" "$elapsed" "$throughput" "$run_status"
    done
  done
done

SUMMARY_MD="$OUT_DIR/summary.md"
{
  echo "# MFNet Throughput Autotune Summary"
  echo
  echo "- Dataset: $SSRS_DATASET"
  echo "- Data root: $SSRS_DATA_ROOT"
  echo "- Epochs per trial: $TUNE_EPOCHS"
  echo "- Steps per epoch: $TUNE_EPOCH_STEPS"
  echo
  echo "## Raw Results"
  echo
  echo '| workers | micro_bs | batch_size | samples | elapsed_sec | throughput(samples/s) | status |'
  echo '|---:|---:|---:|---:|---:|---:|---|'
  tail -n +2 "$RESULT_CSV" | awk -F',' '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $6, $7, $8, $9}'
  echo
  echo "## Best Trial By Wall Time"
  echo
  best_wall_line="$(tail -n +2 "$RESULT_CSV" | awk -F',' '$9=="ok" {print}' | sort -t',' -k7,7n | head -n 1)"
  if [[ -n "$best_wall_line" ]]; then
    wall_workers="$(echo "$best_wall_line" | awk -F',' '{print $1}')"
    wall_micro="$(echo "$best_wall_line" | awk -F',' '{print $2}')"
    wall_batch="$(echo "$best_wall_line" | awk -F',' '{print $3}')"
    wall_elapsed="$(echo "$best_wall_line" | awk -F',' '{print $7}')"
    wall_tput="$(echo "$best_wall_line" | awk -F',' '{print $8}')"
    echo "- workers: $wall_workers"
    echo "- micro_bs: $wall_micro"
    echo "- batch_size: $wall_batch"
    echo "- elapsed_sec: $wall_elapsed"
    echo "- throughput(samples/s): $wall_tput"

    echo
    echo "## Best Trial By Throughput"
    echo
    best_tput_line="$(tail -n +2 "$RESULT_CSV" | awk -F',' '$9=="ok" {print}' | sort -t',' -k8,8nr | head -n 1)"
    tput_workers="$(echo "$best_tput_line" | awk -F',' '{print $1}')"
    tput_micro="$(echo "$best_tput_line" | awk -F',' '{print $2}')"
    tput_batch="$(echo "$best_tput_line" | awk -F',' '{print $3}')"
    tput_elapsed="$(echo "$best_tput_line" | awk -F',' '{print $7}')"
    tput_value="$(echo "$best_tput_line" | awk -F',' '{print $8}')"
    echo "- workers: $tput_workers"
    echo "- micro_bs: $tput_micro"
    echo "- batch_size: $tput_batch"
    echo "- elapsed_sec: $tput_elapsed"
    echo "- throughput(samples/s): $tput_value"

    echo
    echo "### Recommended Exports"
    echo
    echo '```bash'
    echo "# Throughput-priority recommendation"
    echo "export SSRS_NUM_WORKERS=$tput_workers"
    echo "export SSRS_MFNET_MICRO_BS=$tput_micro"
    echo "export SSRS_BATCH_SIZE=$tput_batch"
    echo '```'
  else
    echo "- No successful trial found. Check logs under $LOG_DIR"
  fi
} > "$SUMMARY_MD"

printf "\n[Autotune] Done.\n"
printf "[Autotune] Results CSV: %s\n" "$RESULT_CSV"
printf "[Autotune] Summary MD:  %s\n" "$SUMMARY_MD"
printf "[Autotune] Logs dir:    %s\n" "$LOG_DIR"
