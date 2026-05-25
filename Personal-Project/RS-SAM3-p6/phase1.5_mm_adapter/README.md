# Plan6 Phase 1.5: Online Crops + LoRA/MMAdapter

This phase extends Phase 1 without changing the final evaluation protocol.
It keeps SAM3 frozen, keeps in-ViT RGB/DSM MMAdapter fusion, and adds:

- MFNet-style online random 256x256 crops instead of fixed pre-extracted train windows.
- LoRA inside frozen SAM3 ViT block attention/MLP projections.
- Safe Phase 1 checkpoint initialization; LoRA starts as a no-op.

## Files

| File | Purpose |
|------|---------|
| `dataset_online.py` | Caches full tiles and samples fresh RGB/DSM/label crops each epoch. |
| `mm_adapter_vit.py` | MMAdapter plus LoRA wrappers for `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`. |
| `model_phase1.py` | Builds SAM3 + MMAdapter + LoRA + MFNet decoder. |
| `train_phase1_5.py` | Main Phase 1.5 training entry. |
| `eval_mfnet_protocol.py` | 256x256 sliding-window evaluation; auto-detects LoRA/full-attn args from checkpoint. |

## LoRA Guardrails

Previous standalone LoRA experiments underperformed because LoRA only saw RGB
features and DSM was fused outside the ViT. Phase 1.5 avoids that by keeping
MMAdapter as the primary cross-modal path. LoRA is injected only after DSM has
entered the ViT through MMAdapter.

When DSM uses full attention, LoRA is temporarily disabled on the DSM attention
pass. This prevents one LoRA branch from becoming a shared RGB/DSM shortcut.

## Train From Phase 1 Best

```bash
cd /root/Mynet
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python RS-SAM3-p6/phase1.5_mm_adapter/train_phase1_5.py \
  --dataset vaihingen \
  --epochs 20 \
  --batch 2 \
  --epoch-steps 1000 \
  --lr 1e-4 \
  --dsm-lr 5e-5 \
  --lora-lr 5e-5 \
  --dsm-attn-mode full \
  --checkpoint-attn \
  --lora-rank 8 \
  --lora-alpha 16 \
  --init-from /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

## Evaluate

```bash
python RS-SAM3-p6/phase1.5_mm_adapter/eval_mfnet_protocol.py \
  --dataset vaihingen \
  --checkpoint <run_dir>/best_model.pt
```

Final comparison must use the saved 256x256 sliding-window result, not crop
validation mIoU.
