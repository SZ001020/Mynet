#!/bin/bash
# Run F0→F1→F2→F3 chain automatically
# Usage: bash run_chain.sh
set -e

TRAIN_F0="python /root/Mynet/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3/train_f0.py --dataset vaihingen --epochs 12 --batch 2 --epoch-steps 1000 --lr 1e-4 --seed 42 --output /root/autodl-tmp/runs"
TRAIN_F1="python /root/Mynet/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion/train_f1.py --dataset vaihingen --epochs 8 --batch 2 --epoch-steps 1000 --lr 5e-5 --seed 42 --dsm-attn-mode full --checkpoint-attn --use-lora --output /root/autodl-tmp/runs"
TRAIN_F2="python /root/Mynet/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion/train_f1.py --dataset vaihingen --epochs 8 --batch 2 --epoch-steps 1000 --lr 5e-5 --seed 42 --dsm-attn-mode full --checkpoint-attn --output /root/autodl-tmp/runs"
TRAIN_F3="python /root/Mynet/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion/train_f1.py --dataset vaihingen --epochs 8 --batch 2 --epoch-steps 1000 --lr 5e-5 --seed 42 --dsm-attn-mode full --checkpoint-attn --use-prompt --output /root/autodl-tmp/runs"

F0_RUN=""
F1_RUN=""
F2_RUN=""

echo "=== Plan6 Phase4: F0 → F1 → F2 → F3 chain ==="
echo ""

# ── F0 ──
echo "[F0] Starting training..."
$TRAIN_F0 2>&1 | tee /tmp/phase4_f0.log
F0_RUN=$(ls -td /root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_* | head -1)
F0_BEST="$F0_RUN/best_model.pt"
echo "[F0] Done. Best at: $F0_BEST"
echo ""

# ── F1 ──
echo "[F1] Starting training (init from F0 best)..."
$TRAIN_F1 --init-from "$F0_BEST" 2>&1 | tee /tmp/phase4_f1.log
F1_RUN=$(ls -td /root/autodl-tmp/runs/plan6_phase4_lora_vaihingen_* | head -1)
F1_BEST="$F1_RUN/best_model.pt"
echo "[F1] Done. Best at: $F1_BEST"
echo ""

# ── F2 ──
echo "[F2] Starting training (init from F1 best)..."
$TRAIN_F2 --init-from "$F1_BEST" 2>&1 | tee /tmp/phase4_f2.log
F2_RUN=$(ls -td /root/autodl-tmp/runs/plan6_phase4_frozen_vaihingen_* | head -1)
F2_BEST="$F2_RUN/best_model.pt"
echo "[F2] Done. Best at: $F2_BEST"
echo ""

# ── F3 ──
echo "[F3] Starting training (init from F2 best)..."
$TRAIN_F3 --init-from "$F2_BEST" 2>&1 | tee /tmp/phase4_f3.log
F3_RUN=$(ls -td /root/autodl-tmp/runs/plan6_phase4_prompt_vaihingen_* | head -1)
echo "[F3] Done."
echo ""

echo "=== Phase4 chain complete ==="
echo "F0: $F0_BEST"
echo "F1: $F1_BEST"
echo "F2: $F2_BEST"
echo "F3: $F3_RUN/best_model.pt"
