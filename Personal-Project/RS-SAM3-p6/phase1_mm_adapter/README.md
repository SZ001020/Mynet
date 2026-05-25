# Plan6 Phase 1: In-ViT MMAdapter

This phase implements MFNet-style RGB/DSM interaction inside SAM3 ViTDet blocks.
SAM3 original weights stay frozen; only MMAdapters, the DSM token encoder, and
the MFNet decoder are trainable.

Two DSM token-update modes are supported:

| Mode | Flag | Use case |
|------|------|----------|
| DSM-lite adapter | `--dsm-attn-mode adapter` | Lower memory baseline; DSM stream uses adapter updates instead of a second full SAM3 attention pass. |
| Full DSM self-attention | `--dsm-attn-mode full --checkpoint-attn` | More faithful MFNet-style dual stream; DSM tokens run through frozen SAM3 attention with recomputation to reduce memory. |

## Files

| File | Purpose |
|------|---------|
| `mm_adapter_vit.py` | Replaces SAM3 ViT blocks with RGB/DSM MMAdapter wrappers. |
| `model_phase1.py` | Builds SAM3 + MMAdapter + MFNet decoder. |
| `train_phase1.py` | Trains Phase 1 on Vaihingen or Potsdam 256x256 windows. |
| `dataset_adapter.py` | MFNet split constants and label palette conversion. |
| `mfnet_decoder.py` | MFNet decoder and pyramid utilities. |
| `structure_loss.py` | Edge-weighted BCE + weighted IoU loss. |

## Train

```bash
cd /root/Mynet
python RS-SAM3-p6/phase1_mm_adapter/train_phase1.py \
  --dataset vaihingen \
  --epochs 20 \
  --batch 2 \
  --lr 1e-4 \
  --dsm-lr 5e-5
```

Full DSM self-attention on GPU 1:

```bash
cd /root/Mynet
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python RS-SAM3-p6/phase1_mm_adapter/train_phase1.py \
  --dataset vaihingen \
  --epochs 20 \
  --batch 2 \
  --lr 1e-4 \
  --dsm-lr 5e-5 \
  --dsm-attn-mode full \
  --checkpoint-attn \
  --output /root/autodl-tmp/runs
```

Outputs are written to `/root/autodl-tmp/runs/plan6_phase1_mm_adapter_*` and
include `best_model.pt`, `history.json`, `metrics.json`, and `config.json`.

## Evaluation Contract

Training-time validation reports `avg_oa`, `avg_miou`, `per_class_iou`, and
`per_class_oa`. Final model comparison must still use the repository-wide
256x256 sliding-window evaluation protocol from `plan6.md`.

Current evaluated Phase 1 baseline:

```bash
python RS-SAM3-p6/phase1_mm_adapter/eval_mfnet_protocol.py \
  --dataset vaihingen \
  --checkpoint /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_161834/best_model.pt
```

Result: `OA=87.02`, `mIoU=74.81`; per-class IoU is `road=76.17`,
`building=86.24`, `grass=60.43`, `tree=76.43`, `car=74.81`.

Current full DSM self-attention run:

| Run | Config | Early best |
|-----|--------|------------|
| `/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720` | batch 2, GPU1, full DSM self-attention + checkpoint | `76.86` crop mIoU at epoch 10 |

Full DSM self-attention final 256x256 evaluation:

```bash
python RS-SAM3-p6/phase1_mm_adapter/eval_mfnet_protocol.py \
  --dataset vaihingen \
  --checkpoint /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt \
  --output /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/eval_256_vaihingen_plan6_phase1_full.json
```

Result: `OA=87.27`, `mIoU=75.72`; per-class IoU is `road=76.08`,
`building=87.03`, `grass=61.31`, `tree=76.99`, `car=77.21`.
