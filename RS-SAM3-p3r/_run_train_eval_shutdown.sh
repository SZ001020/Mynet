#!/bin/bash
# Plan3 p3r: Train → Eval → Shutdown
set -e
cd /root/Mynet/RS-SAM3-p3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "======================================================"
echo "Phase 1: Training (gated DSM + UNetFormer)"
echo "======================================================"
python -u train_dual.py --model dsm --dataset vaihingen --epochs 20 --batch 4
echo "Training done."

echo ""
echo "======================================================"
echo "Phase 2: 256² Sliding Window Evaluation"
echo "======================================================"
# Find latest checkpoint
CKPT=$(ls -t /root/autodl-tmp/runs/plan3_p3r_dsm_*/best_model.pt 2>/dev/null | head -1)
echo "Checkpoint: $CKPT"

python -u _eval_quick.py

echo ""
echo "======================================================"
echo "All done. Shutting down..."
echo "======================================================"
sleep 5
shutdown -h now
