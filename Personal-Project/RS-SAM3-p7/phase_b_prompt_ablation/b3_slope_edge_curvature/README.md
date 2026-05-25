# Plan7-B3: DSM Slope/Edge/Curvature Prompt

This ablation tests whether adding a smooth curvature/roughness prior improves
on Plan7-A's DSM slope+edge prompt. It keeps the Plan7-A architecture and
training protocol unchanged except for the prompt channels.

Prompt channels:

```text
slope = Sobel magnitude
edge = abs(Laplacian DSM)
curvature = local average of edge
```

## Train

```bash
cd /root/Mynet
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
  --checkpoint-attn
```

Default initialization is the Plan6 Phase1 full DSM self-attention best
checkpoint. This keeps B3 comparable with Plan7-A as a formal prompt ablation.

```text
/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

If initializing from Plan7-A best for a faster continuation run, record it as a
continuation experiment rather than a formal B-stage ablation.

## Evaluate

```bash
python RS-SAM3-p7/phase_b_prompt_ablation/b3_slope_edge_curvature/eval.py \
  --dataset vaihingen \
  --checkpoint <run_dir>/best_model.pt
```
