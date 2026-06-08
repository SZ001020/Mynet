#!/bin/bash
# Plan11: Sequential experiment runner
# Runs all P0+P1 experiments one after another on Vaihingen
set -e
cd /root/Mynet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Plan11 Experiment Pipeline"
echo "Started: $(date)"
echo "=============================================="

# === P11-A: veg boundary loss (from Plan7-A init) ===
echo ""
echo "=== [1/5] P11-A: veg boundary loss ==="
echo "Started: $(date)"
python Personal-Project/RS-SAM3-p11/phase_a_veg_boundary_loss/train.py \
  --dataset vaihingen --epochs 15 --batch 2 --epoch-steps 1000 \
  --resolution 1008 --dsm-attn-mode full --checkpoint-attn \
  --init-from /root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_225309/best_model.pt \
  --veg-boundary-weight 3.0
P11A_DIR=$(ls -td /root/autodl-tmp/runs/plan11_a_veg_boundary_loss_vaihingen_* | head -1)
echo "P11-A output: $P11A_DIR"
python Personal-Project/RS-SAM3-p11/eval.py \
  --checkpoint "$P11A_DIR/best_model.pt" --dataset vaihingen

# === P11-B: nDSM baseline (from scratch) ===
echo ""
echo "=== [2/5] P11-B: nDSM baseline (from scratch) ==="
echo "Started: $(date)"
python Personal-Project/RS-SAM3-p11/phase_b_ndsm/train.py \
  --dataset vaihingen --epochs 15 --batch 2 --epoch-steps 1000 \
  --resolution 1008 --dsm-attn-mode full --checkpoint-attn --seed 42
P11B_DIR=$(ls -td /root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_* | head -1)
echo "P11-B output: $P11B_DIR"
python Personal-Project/RS-SAM3-p11/eval.py \
  --checkpoint "$P11B_DIR/best_model.pt" --dataset vaihingen --ndsm

# === P11-C: combined nDSM + veg boundary (from scratch) ===
echo ""
echo "=== [3/5] P11-C: combined nDSM + veg boundary ==="
echo "Started: $(date)"
python Personal-Project/RS-SAM3-p11/phase_c_combined/train.py \
  --dataset vaihingen --epochs 15 --batch 2 --epoch-steps 1000 \
  --resolution 1008 --dsm-attn-mode full --checkpoint-attn --seed 42 \
  --veg-boundary-weight 3.0
P11C_DIR=$(ls -td /root/autodl-tmp/runs/plan11_c_combined_vaihingen_* | head -1)
echo "P11-C output: $P11C_DIR"
python Personal-Project/RS-SAM3-p11/eval.py \
  --checkpoint "$P11C_DIR/best_model.pt" --dataset vaihingen --ndsm

# === P11-D: combined + augmentation ===
echo ""
echo "=== [4/5] P11-D: combined + augmentation ==="
echo "Started: $(date)"
python Personal-Project/RS-SAM3-p11/phase_d_augmentation/train.py \
  --dataset vaihingen --epochs 15 --batch 2 --epoch-steps 1000 \
  --resolution 1008 --dsm-attn-mode full --checkpoint-attn --seed 42 \
  --veg-boundary-weight 3.0 --aug-color-jitter 0.2 --aug-blur-prob 0.3
P11D_DIR=$(ls -td /root/autodl-tmp/runs/plan11_d_aug_vaihingen_* | head -1)
echo "P11-D output: $P11D_DIR"
python Personal-Project/RS-SAM3-p11/eval.py \
  --checkpoint "$P11D_DIR/best_model.pt" --dataset vaihingen --ndsm

# === P11-E: adapter ablation ===
echo ""
echo "=== [5/5] P11-E: adapter ablation (bottleneck=8, global only) ==="
echo "Started: $(date)"
python Personal-Project/RS-SAM3-p11/phase_e_adapter_ablation/train.py \
  --dataset vaihingen --epochs 15 --batch 2 --epoch-steps 1000 \
  --resolution 1008 --dsm-attn-mode full --checkpoint-attn --seed 42 \
  --adapter-bottleneck 8 --adapter-placement global
P11E_DIR=$(ls -td /root/autodl-tmp/runs/plan11_e_adapter_vaihingen_* | head -1)
echo "P11-E output: $P11E_DIR"
python Personal-Project/RS-SAM3-p11/eval.py \
  --checkpoint "$P11E_DIR/best_model.pt" --dataset vaihingen --ndsm

echo ""
echo "=============================================="
echo "Plan11 ALL DONE: $(date)"
echo "=============================================="
