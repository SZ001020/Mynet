#!/bin/bash
# Plan8 fix-up: re-run CA-A and AB-A with memory fixes
# CTRL already done (OA=86.33%, mIoU=74.77%)
# CA-A: flash attention fix + checkpoint_attn, batch=2
# AB-A: batch=1 (manual attention OOM at batch=2)
set -e

LOG_DIR="/root/autodl-tmp/runs"
echo "=== Plan8 Fix-up (CA-A + AB-A) ==="
echo "Start: $(date)"
echo "CTRL already done: mIoU=74.77% (256^2 eval)"

# ── STEP 1: CA-A (flash attention, batch=2) ─────────────────
echo ""
echo ">>> [1/2] CA-A cross-attention (flash attn, batch=2, checkpoint_attn)"
cd /root/Mynet/RS-SAM3-p8/chain1_cross_attn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python train_ca_a.py --dataset vaihingen --epochs 8 --batch 2 --lr 5e-5 \
  --dsm-attn-mode full --checkpoint-attn
CA_CKPT=$(ls -t ${LOG_DIR}/plan8_ca_a_*/best_model.pt 2>/dev/null | head -1)
echo "CA-A best: ${CA_CKPT}"

echo "--- CA-A eval ---"
python eval.py --checkpoint "${CA_CKPT}" --dataset vaihingen
echo "CA-A eval done"

# ── STEP 2: AB-A (batch=1 due to manual attn OOM) ───────────
echo ""
echo ">>> [2/2] AB-A attention bias (batch=1, checkpoint_attn)"
cd /root/Mynet/RS-SAM3-p8/chain2_attn_bias
CUDA_VISIBLE_DEVICES=0 \
  python train_ab_a.py --dataset vaihingen --epochs 8 --batch 1 --lr 5e-5 \
  --dsm-attn-mode full --checkpoint-attn
AB_CKPT=$(ls -t ${LOG_DIR}/plan8_ab_a_*/best_model.pt 2>/dev/null | head -1)
echo "AB-A best: ${AB_CKPT}"

echo "--- AB-A eval ---"
python eval.py --checkpoint "${AB_CKPT}" --dataset vaihingen
echo "AB-A eval done"

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "=== Plan8 Fix-up Complete ==="
echo "End: $(date)"
echo ""
echo "CTRL: OA=86.33% mIoU=74.77% (existing)"
python3 -c "
import json, os
for name, pat in [('CA-A','plan8_ca_a_*'),('AB-A','plan8_ab_a_*')]:
    dirs = sorted([d for d in os.listdir('${LOG_DIR}') if d.startswith(pat[:10])], reverse=True)
    if dirs:
        jf = os.path.join('${LOG_DIR}', dirs[0], 'eval_256_vaihingen.json')
        jf2 = os.path.join('${LOG_DIR}', dirs[0], 'eval_256_vaihingen_plan8_ca.json')
        jf3 = os.path.join('${LOG_DIR}', dirs[0], 'eval_256_vaihingen_plan8_ab.json')
        for f in [jf, jf2, jf3]:
            if os.path.exists(f):
                d = json.load(open(f))
                print(f'{name}: OA={d.get(\"avg_oa\",0):.2f}  mIoU={d.get(\"avg_miou\",0):.2f}')
                break
        else:
            print(f'{name}: eval JSON not found')
" 2>/dev/null
