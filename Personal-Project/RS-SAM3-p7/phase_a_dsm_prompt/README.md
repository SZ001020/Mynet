# Plan7-A: DSM Edge/Slope Prompt

This phase tests one minimal question: can DSM structure prompts improve the
Plan6 Phase1 full-attention baseline (`75.72` final 256x256 mIoU)?

It adds a DSM edge/slope prompt branch to the in-ViT RGB/DSM MMAdapter and keeps
the SAM3 backbone frozen.

## Train

```bash
cd /root/Mynet
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python RS-SAM3-p7/phase_a_dsm_prompt/train_a.py \
  --dataset vaihingen \
  --epochs 8 \
  --batch 2 \
  --epoch-steps 1000 \
  --lr 5e-5 \
  --dsm-lr 2.5e-5 \
  --prompt-lr 2.5e-5 \
  --resolution 1008 \
  --dsm-attn-mode full \
  --checkpoint-attn \
  --init-from /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

## Evaluate

```bash
python RS-SAM3-p7/phase_a_dsm_prompt/eval.py \
  --dataset vaihingen \
  --checkpoint <run_dir>/best_model.pt
```

Only final 256x256 sliding-window results should be compared across plans.

For speed/scale ablation, run a short 768-resolution smoke experiment:

```bash
python RS-SAM3-p7/phase_a_dsm_prompt/train_a.py \
  --dataset vaihingen \
  --epochs 2 \
  --batch 2 \
  --epoch-steps 500 \
  --resolution 768 \
  --lr 5e-5 \
  --dsm-lr 2.5e-5 \
  --prompt-lr 2.5e-5 \
  --dsm-attn-mode full \
  --checkpoint-attn \
  --init-from /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```
