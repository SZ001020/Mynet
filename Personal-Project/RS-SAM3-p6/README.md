# RS-SAM3-p6

Plan6 experiment workspace. Each phase is isolated in its own subdirectory so code,
checkpoints, logs, and ablations do not conflict across training routes.

See `/root/Mynet/plan6.md` for the full training plan and fixed evaluation protocol.

## Directory Map

| Directory | Phase | Purpose |
|-----------|-------|---------|
| `phase1_mm_adapter/` | Phase 1 | MFNet-style in-ViT RGB/DSM MMAdapter for SAM3 ViTDet. |
| `phase1.5_mm_adapter/` | Phase 1.5 | Online random crops plus LoRA/MMAdapter cooperation. |
| `phase2_dsm_frequency_prompt/` | Phase 2 | DSM high-pass, slope, hillshade, edge, and RS-token prompt injection. |
| `phase3_prototype_hard_negative/` | Phase 3 | Prototype matching and hard-negative regularization for confused classes. |
| `phase4_boundary_object_loss/` | Phase 4 | SAM3/SAM_RS-style boundary and object auxiliary supervision. |
| `phase5_pretrain_joint/` | Phase 5 | Joint Vaihingen+Potsdam training and LoveDA pretrain to ISPRS finetune. |

## Shared Rules

- Keep the MFNet train/test splits unchanged.
- Use 5 foreground classes only: road, building, grass, tree, car.
- Final evaluation must use 256x256 sliding window with stride 128.
- Every evaluation must output `avg_oa`, `avg_miou`, `per_class_iou`, and `per_class_oa`.
- Save run outputs under `/root/autodl-tmp/runs/plan6_*`.

## Current Status

Phase 1 is implemented and has two DSM attention modes:

| Mode | Command flag | Status |
|------|--------------|--------|
| DSM-lite adapter stream | `--dsm-attn-mode adapter` | Completed. Best evaluated checkpoint: `/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_161834/best_model.pt`. |
| Full DSM self-attention | `--dsm-attn-mode full --checkpoint-attn` | Completed. Run: `/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720`. |

Best completed 256x256 sliding-window result so far:

| Run | OA | mIoU | Notes |
|-----|----|------|-------|
| Phase1 DSM-lite | 87.02 | 74.81 | First evaluated Plan6 checkpoint. |
| Phase1 full DSM self-attention | 87.27 | 75.72 | Current strongest evaluated Plan6 checkpoint. |

The full DSM self-attention run reached `76.86` crop validation mIoU at epoch 10
and improves the final 256x256 result by `+0.91pp` over DSM-lite.
