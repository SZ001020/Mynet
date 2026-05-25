#!/usr/bin/env bash
set -euo pipefail

cd /root/Mynet

TS="$(date +%Y%m%d_%H%M%S)"
LOG="/root/autodl-tmp/runs/plan7_b_all_${TS}.log"
INIT="/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt"

echo "Plan7-B batch run started at ${TS}" | tee -a "${LOG}"
echo "Formal ablation init_from=${INIT}" | tee -a "${LOG}"

run_b1() {
  echo "" | tee -a "${LOG}"
  echo "=== Plan7-B1 slope_only train ===" | tee -a "${LOG}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python RS-SAM3-p7/phase_b_prompt_ablation/b1_slope_only/train_b1.py \
    --dataset vaihingen \
    --epochs 8 \
    --batch 2 \
    --epoch-steps 1000 \
    --resolution 1008 \
    --lr 5e-5 \
    --dsm-lr 2.5e-5 \
    --prompt-lr 2.5e-5 \
    --dsm-attn-mode full \
    --checkpoint-attn \
    --init-from "${INIT}" \
    2>&1 | tee -a "${LOG}"

  B1_DIR="$(ls -td /root/autodl-tmp/runs/plan7_phase_b1_slope_only_vaihingen_* | head -1)"
  echo "=== Plan7-B1 eval: ${B1_DIR} ===" | tee -a "${LOG}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python RS-SAM3-p7/phase_b_prompt_ablation/b1_slope_only/eval.py \
    --dataset vaihingen \
    --checkpoint "${B1_DIR}/best_model.pt" \
    --output "${B1_DIR}/eval_256_vaihingen_plan7_b1_global.json" \
    2>&1 | tee -a "${LOG}"
}

run_b3() {
  echo "" | tee -a "${LOG}"
  echo "=== Plan7-B3 slope_edge_curvature train ===" | tee -a "${LOG}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python RS-SAM3-p7/phase_b_prompt_ablation/b3_slope_edge_curvature/train_b3.py \
    --dataset vaihingen \
    --epochs 8 \
    --batch 2 \
    --epoch-steps 1000 \
    --resolution 1008 \
    --lr 5e-5 \
    --dsm-lr 2.5e-5 \
    --prompt-lr 2.5e-5 \
    --dsm-attn-mode full \
    --checkpoint-attn \
    --init-from "${INIT}" \
    2>&1 | tee -a "${LOG}"

  B3_DIR="$(ls -td /root/autodl-tmp/runs/plan7_phase_b3_slope_edge_curvature_vaihingen_* | head -1)"
  echo "=== Plan7-B3 eval: ${B3_DIR} ===" | tee -a "${LOG}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python RS-SAM3-p7/phase_b_prompt_ablation/b3_slope_edge_curvature/eval.py \
    --dataset vaihingen \
    --checkpoint "${B3_DIR}/best_model.pt" \
    --output "${B3_DIR}/eval_256_vaihingen_plan7_b3_global.json" \
    2>&1 | tee -a "${LOG}"
}

run_b1
run_b3

echo "" | tee -a "${LOG}"
echo "Plan7-B batch run finished at $(date +%Y%m%d_%H%M%S). Shutting down now." | tee -a "${LOG}"
sync
shutdown -h now
