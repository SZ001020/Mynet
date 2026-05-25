#!/bin/bash
# Plan8 sequential training + eval: CTRL → CA-A → AB-A
# Total training: ~9h; eval: ~30min per model

set -e

LOG_DIR="/root/autodl-tmp/runs"
echo "=== Plan8 Sequential Pipeline (Train + Eval) ==="
echo "Start: $(date)"

# ── STEP 1: CTRL Baseline ──────────────────────────────────
echo ">>> [1/3] CTRL baseline"
cd /root/Mynet/RS-SAM3-p8/ctrl_baseline
CUDA_VISIBLE_DEVICES=0 python train_ctrl.py \
  --dataset vaihingen --epochs 8 --batch 2 --lr 5e-5 --dsm-attn-mode full
CTRL_CKPT=$(ls -t ${LOG_DIR}/plan8_ctrl_*/best_model.pt 2>/dev/null | head -1)
echo "CTRL best: ${CTRL_CKPT}"

# ── STEP 2: CA-A Cross-Attention ───────────────────────────
echo ">>> [2/3] CA-A cross-attention"
cd /root/Mynet/RS-SAM3-p8/chain1_cross_attn
CUDA_VISIBLE_DEVICES=0 python train_ca_a.py \
  --dataset vaihingen --epochs 8 --batch 2 --lr 5e-5 --dsm-attn-mode full
CA_CKPT=$(ls -t ${LOG_DIR}/plan8_ca_a_*/best_model.pt 2>/dev/null | head -1)
echo "CA-A best: ${CA_CKPT}"

# ── STEP 3: AB-A Attention Bias ────────────────────────────
echo ">>> [3/3] AB-A attention bias"
cd /root/Mynet/RS-SAM3-p8/chain2_attn_bias
CUDA_VISIBLE_DEVICES=0 python train_ab_a.py \
  --dataset vaihingen --epochs 8 --batch 2 --lr 5e-5 --dsm-attn-mode full
AB_CKPT=$(ls -t ${LOG_DIR}/plan8_ab_a_*/best_model.pt 2>/dev/null | head -1)
echo "AB-A best: ${AB_CKPT}"

# ── Eval: 三组并行评估 ─────────────────────────────────────
echo ""
echo "=== Evaluation ==="

echo "--- CTRL eval ---"
cd /root/Mynet/RS-SAM3-p8/ctrl_baseline
python eval.py --checkpoint "${CTRL_CKPT}" --dataset vaihingen
CTRL_EVAL=$(ls -t ${LOG_DIR}/plan8_ctrl_*/eval_256_vaihingen.json 2>/dev/null | head -1)

echo "--- CA-A eval ---"
cd /root/Mynet/RS-SAM3-p8/chain1_cross_attn
python eval.py --checkpoint "${CA_CKPT}" --dataset vaihingen
CA_EVAL=$(ls -t ${LOG_DIR}/plan8_ca_a_*/eval_256_vaihingen.json 2>/dev/null | head -1)

echo "--- AB-A eval ---"
cd /root/Mynet/RS-SAM3-p8/chain2_attn_bias
python eval.py --checkpoint "${AB_CKPT}" --dataset vaihingen
AB_EVAL=$(ls -t ${LOG_DIR}/plan8_ab_a_*/eval_256_vaihingen.json 2>/dev/null | head -1)

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "=== Plan8 Complete ==="
echo "End: $(date)"
echo ""
echo "Results:"
echo "  CTRL: ${CTRL_EVAL}"
python3 -c "
import json
for name, path in [('CTRL','${CTRL_EVAL}'),('CA-A','${CA_EVAL}'),('AB-A','${AB_EVAL}')]:
    d=json.load(open(path))
    oa=d.get('avg_oa','?'); miou=d.get('avg_miou','?')
    print(f'  {name}: OA={oa:.2f}, mIoU={miou:.2f}')
" 2>/dev/null || echo "(eval JSON paths for manual inspection above)"
