#!/usr/bin/env bash
set -euo pipefail

cd /root/Mynet

PY=Personal-Project/RS-SAM3-p13/phase_g_strict_veg_texture/train.py
BASE_ARGS=(--dataset vaihingen --epochs 10 --batch 2 --epoch-steps 1000 --seed 42 --val-every 10)

echo "=== P13-G1: oracle mask + RGB multi-scale texture ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$PY" \
  --experiment g1_oracle_rgb \
  --select-mask oracle \
  "${BASE_ARGS[@]}"

echo "=== P13-G2: pred mask + RGB multi-scale texture ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$PY" \
  --experiment g2_pred_rgb \
  --select-mask pred \
  "${BASE_ARGS[@]}"

echo "=== P13-G3: oracle mask + RGB multi-scale texture + nDSM roughness ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$PY" \
  --experiment g3_oracle_rgb_ndsmrough \
  --select-mask oracle \
  --use-ndsm-roughness \
  "${BASE_ARGS[@]}"

echo "=== P13-G4: pred mask + RGB multi-scale texture + nDSM roughness ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$PY" \
  --experiment g4_pred_rgb_ndsmrough \
  --select-mask pred \
  --use-ndsm-roughness \
  "${BASE_ARGS[@]}"

echo "=== P13-G5: pred mask + RGB multi-scale texture + nDSM roughness + DiceCE ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$PY" \
  --experiment g5_pred_rgb_ndsmrough_dicece \
  --select-mask pred \
  --use-ndsm-roughness \
  --loss-type dicece \
  "${BASE_ARGS[@]}"

echo "All P13-G experiments finished."
