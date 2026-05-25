# Plan7-B1: DSM Slope-Only Prompt

This ablation tests whether Plan7-A's gain mainly comes from DSM slope rather
than Laplacian edge information. It keeps the Plan7-A architecture and training
protocol unchanged except for the prompt channels.

Baseline for comparison:

```text
Plan7-A latest: 256 sliding mIoU = 76.27
```

## Train

```bash
cd /root/Mynet
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
  --checkpoint-attn
```

Default initialization is the Plan6 Phase1 full DSM self-attention best
checkpoint. This keeps B1 comparable with Plan7-A as a formal prompt ablation.

```text
/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

If initializing from Plan7-A best for a faster continuation run, record it as a
continuation experiment rather than a formal B-stage ablation.

## Evaluate

```bash
python RS-SAM3-p7/phase_b_prompt_ablation/b1_slope_only/eval.py \
  --dataset vaihingen \
  --checkpoint <run_dir>/best_model.pt
```

Only the 256x256 sliding-window result should be used for Plan7-B decisions.
