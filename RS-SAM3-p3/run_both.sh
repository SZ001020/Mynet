#!/bin/bash
# Run Plan3 Route A training: Potsdam then Vaihingen (MFNet splits)
# Sequential because SAM3 takes ~31GB GPU (only 1 model fits on RTX 5090)

set -e
cd /root/Mynet/RS-SAM3-p3
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Phase 1: Potsdam training (MFNet split)"
echo "=========================================="
python -u train_adapter.py --dataset potsdam --epochs 50 --batch 8 --lr 1e-4 --val-every 5 --num-workers 2

echo ""
echo "=========================================="
echo "Phase 2: Vaihingen training (MFNet split)"
echo "=========================================="
python -u train_adapter.py --dataset vaihingen --epochs 50 --batch 8 --lr 1e-4 --val-every 5 --num-workers 2

echo ""
echo "Both trainings complete!"
