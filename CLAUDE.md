# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hardware & Environment

- GPU: RTX 5090 32GB (CUDA 12.8), PyTorch 2.7.0, bf16 + FA3
- RAM: 754 GB
- Data disk: `/root/autodl-tmp` (100G) — checkpoints, run outputs
- Public disk: `/autodl-pub` (14T) — model weights, dataset backups

## Active Research

This is the SSRS remote sensing segmentation repo. Active branch: `segearthov3`, working through 9 research plans (plan.md → plan9.md). The task is frozen-backbone SAM3 semantic segmentation on ISPRS Vaihingen/Potsdam with DSM (digital surface model) fusion.

**Core architecture**: SAM3 ViTDet (frozen) + in-ViT MMAdapter (RGB/DSM dual-stream at every ViT block) + MFNetDecoder.

**Current best (2026-05-19)**:

| Model | mIoU | OA | Trainable | Eval |
|-------|------|-----|-----------|------|
| **Plan7-A: DSM edge/slope prompt** | **77.55%** | 88.20% | ~5M | 软 logit |
| Plan7-A (old eval) | 77.14% | 87.75% | ~5M | per-patch argmax |
| Phase4 F0: shared + LoRA + SEFusion | 77.34% | 88.23% | ~7.2M | 软 logit |
| Plan6 Phase1: full DSM attention | 77.38% | 87.83% | ~5M | 软 logit |
| Plan6 Phase1 (old eval) | 76.57% | 87.27% | ~5M | per-patch argmax |
| MFNet Frozen SAM1 (reference) | 75.11% | 88.01% | LoRA | SAM1 |

**Phase 4 key finding (2026-05-19)**: LoRA on SAM3 ViTDet (rank=8) is the dominant factor (+1.95pp from F0'→F0'+L), comparable in importance to in-ViT MMAdapter. Shared encoder + LoRA + 1×SEFusion (F0: 77.34%) is competitive but slightly below Plan7-A (77.55%, same soft-logit eval, -0.21pp). Frozen + LoRA + 4×SEFusion (F0'+L: 77.24%) is close behind. LoRA + simple architecture ≈ frozen + complex adapter architecture. The value of Phase 4 is the fair controlled comparison, not a new SOTA.

**Plan9 Phase A (2026-05-26) — NEGATIVE**: LoRA + DSM edge/slope input-level fusion. Edge/slope concatenated as extra DSM channels → Conv1×1 → SAM3 encoder. P9-A degraded F0'+L from 77.24% to ~76.45% (-0.79pp). Reason: input-level channel concatenation loses structural information in patch_embed; SAM3's RGB-pretrained patch_embed cannot interpret DSM+edge+slope channel semantics. Plan7-A's +0.57pp comes from token-level PromptEncoder + gate injection, not from the edge/slope data itself.

**Plan9 Phase B (2026-05-27) — 规划中**: Token-level PromptEncoder + gate injection on LoRA baseline. Same paradigm as Plan7-A (PromptEncoder → per-block 3-way gate), but replacing MMAdapter with LoRA. Gate directly weights attn outputs (no adapter MLPs), prompt stream uses frozen self-attention. Single experiment P9-B2 from F0'+L (77.24%). Goal: verify whether LoRA + prompt gate can be additive (Plan7-A's prompt contribution was measured on adapter architecture, not LoRA).

**Plan8 (2026-05-16) — NEGATIVE**: Cross-attention (-0.26pp) and DSM attention bias (-0.65pp) in frozen ViT provide no benefit over gate-only baseline. Gate fusion alone is sufficient for RGB-DSM interaction at this scale.

**Phase 3 ablation (2026-05-14)**: in-ViT MMAdapter is the dominant contributor (+6.75pp over late fusion). Gate type (sigmoid vs softmax), loss (CE vs structure_loss), and data loading (fixed vs online crops) each contribute ≤0.2pp. Decoder upgrade from simple to MFNetDecoder adds ~0.7pp.

**Failed directions — do not retry**:

| Direction | Evidence | Δ vs baseline | Plan |
|-----------|----------|---------------|------|
| Cross-attention RGB↔DSM in frozen ViT | CA-A 74.52 vs CTRL 74.77 | -0.26pp | Plan8 |
| DSM-derived attention bias | AB-A 74.12 vs CTRL 74.77 | -0.65pp | Plan8 |
| Full fine-tuning | 72.87 vs frozen 73.54 | degrades | Plan4 |
| LoRA on frozen backbone | 75.57 vs 76.57 | -1.00pp | Plan6 |
| Unfreeze attention layers | 72.51 vs 76.57 | -4.06pp | Plan6 |
| Boundary/object auxiliary loss | — | no gain | Plan5 |
| SAM-HQ residual correction | 77.05 vs 77.14 | -0.09pp | Plan7-C3 |
| Multi-scale TTA | 77.33 vs 77.55 | -0.22pp | Plan7-D1 |
| Curvature/roughness prompt expansion | 76.49 vs 77.14 | -0.65pp | Plan7-B3 |
| Input-level DSM edge/slope channel concat | 76.45 vs 77.24 | -0.79pp | Plan9-A |
| LoRA + input-level edge/slope prompt | 76.45 vs 76.81 | -0.36pp | Plan9 |

**Before proposing any new direction, check this table and the Experiment Lineage below.**

## Experiment Lineage

Every experiment branch is recorded here. Read top-to-bottom to trace the full derivation chain. Each `→` is an experiment; ✅/❌/➖ marks gain / loss / noise-level result. Lines ending in ❌ are closed and should not be retried.

```
Plan6 Phase1 (76.57%) — first in-ViT MMAdapter, no prompt
  │
  ├─→ Plan7-A (+ Sobel/Laplacian prompt) → 77.14% ✅ +0.57pp  [CURRENT BEST]
  ├─→ Plan7-B1 (slope only prompt) → 76.49% ❌ -0.65pp
  ├─→ Plan7-B3 (slope+edge+curvature) → 76.49% ❌ no gain
  ├─→ Plan7-C3 (SAM-HQ residual correction) → 77.05% ➖ -0.09pp
  └─→ Plan7-D1 (multi-scale TTA) → 77.33% ➖ -0.22pp

Plan6 Phase3 (ablation) — all variants from Plan6 Phase1 parent
  ├─→ A0: no adapter → 53.10%
  ├─→ A1: late fusion → 65.86% (+12.76pp)
  ├─→ A3: in-ViT MMAdapter → 72.88% (+6.75pp over A1) ← KEY ABLATION
  ├─→ B1: softmax gate → +0.04pp ➖
  ├─→ C0: fixed windows → noise-level
  └─→ D1: CE loss → noise-level

Plan8 — SAM3 base, seed=42, all from scratch (formal ablation)
  │
  ├─→ CTRL (gate-only, Plan7-A architecture) → 74.77%
  ├─→ CA-A (+ DSM→RGB cross-attn, 4 global blocks) → 74.52% ❌ -0.26pp
  │   └─→ Chain 1 closed: cross-attn in frozen ViT provides no benefit
  └─→ AB-A (+ DSM elevation attn bias, factorized) → 74.12% ❌ -0.65pp
      └─→ Chain 2 closed: attn bias degrades frozen attention

Plan6 Phase4 — MFNet strict comparison, seed=42 (formal ablation)
  │
  ├─→ F0 (shared encoder + LoRA + 1×SEFusion + MFNetDecoder) → 77.34% ✅ [PHASE4 BEST]
  ├─→ F1 (in-ViT MMAdapter + LoRA) → 77.29% ➖ -0.11pp vs F0
  │   └─→ in-ViT adapter provides NO gain over shared+SEFusion when LoRA present
  ├─→ F2 (in-ViT MMAdapter, frozen) → 75.51% ❌ -1.49pp vs F1
  │   └─→ LoRA is the DOMINANT factor: contributes ~1.95pp
  └─→ F3 (in-ViT MMAdapter, frozen + prompt) → 76.52% (+0.65pp vs F2)
      └─→ Prompt gain consistent with Plan7-A (+0.57pp); confirms reproducibility

Plan6 Phase4 子链 — F0' pure frozen baseline (continuation from Phase4)
  │
  ├─→ F0' (SAM3 frozen + 4×SEFusion + MFNetDecoder) → 75.29% [对标 MFNet 75.11%]
  └─→ F0'+L (F0' + LoRA rank=8) → 77.24% ✅ +1.95pp ← LoRA 净贡献

Plan9 — LoRA + DSM edge/slope fusion
  │
  ├─→ Phase A (from F0'+L 77.24%): input-level 3ch DSM → 76.45% ❌ -0.36pp
  │   └─→ Closed: edge/slope only works via token-level gate, not input enrichment
  └─→ Phase B (from F0'+L 77.24%): token-level PromptEncoder + per-block gate (规划中)

Earlier plans (superseded):
  Plan3: Adapter+decoder beats full fine-tuning despite 175× fewer params
  Plan4: Full ViT training degrades; unfreeze harms frozen backbone
  Plan5: Boundary/object auxiliary loss provides no gain
  Plan1-2: Zero-shot baselines; per-class binary paradigm overestimates
```

**Lineage conventions:**
- `formal_ablation`: same parent ckpt, single variable change. Directly comparable.
- `continuation`: loaded from best ckpt, but changed multiple variables. Compare with caution.
- `from scratch`: no inherited weights. Compare only within same seed group.
- All experiments must record `init_from`, `lineage_type`, and `comparison_role` in both `plan{N}.md` and `model_registry.py`.

## Key Architecture

### In-ViT MMAdapter (Plan6/7)

DSM tokens enter SAM3 ViT at every layer alongside RGB tokens, sharing the frozen attention mechanism:

```
RGB → ViT patch embed → RGB tokens (BHWC, 1024-dim)
DSM → CNN encoder → DSM tokens (same spatial grid)

In every ViT block (32 blocks, global attn at [7,15,23,31]):
  RGB attn = frozen_self_attention(RGB)
  DSM attn = frozen_self_attention(DSM)  ← "full" mode reuses same weights
  gate = softmax(learnable_logits) → 3-way fusion (RGB/DSM/prompt)
  output = gate[0]*RGB_adapter + gate[1]*DSM_adapter + gate[2]*prompt_adapter

After ViT: multi-scale features (blocks 8/16/24/32 output via backbone_fpn)
  → Pyramid4Scale (4-scale FPN)
  → MFNetDecoder (3 GLA blocks + FeatureRefinementHead) → 5-class logits
```

Plan7 extends this by adding a third input branch: DSM edge+slope prompt tokens computed from the DSM via Sobel/Laplacian, encoded by a separate PromptEncoder, and injected into each block's 3-way softmax gate.

**Critical implementation detail**: The ViTDet is accessed via `vision_backbone.trunk.blocks` (not `vision_backbone.blocks`). The `vision_backbone` is a `Sam3DualViTDetNeck` wrapping a ViT `trunk` with 32 blocks, each wrapped by `MMAdapterPromptBlock`.

### MFNetDecoder

Input: 4-scale features [1/4, 1/8, 1/16, 1/32] from Pyramid4Scale.
Architecture: GLA (Global-Local Attention) blocks with window attention + relative position bias, WF weighted fusion modules, FeatureRefinementHead with PA+CA dual attention.
Output: [B, 5, H/4, W/4] logits.

## Project Directory Map

Our code lives under `Personal-Project/`. Vendored reference projects under `Reference-Project/`. Plans under `Plan/`.

| Directory | Purpose | Status |
|-----------|---------|--------|
| `Personal-Project/RS-SAM3-p6/phase1_mm_adapter/` | Plan6 Phase1: in-ViT MMAdapter (best 76.57%) | Reference |
| `Personal-Project/RS-SAM3-p6/phase3_ablation/` | Plan6 Phase3: adapter/gate/data/loss/decoder ablation | Done |
| `Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/` | Plan6 Phase4: MFNet strict comparison, LoRA ablation (best F0 77.34%) | Done |
| `Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/` | Plan7-A: DSM edge/slope prompt (best 77.55%) | Reference |
| `Personal-Project/RS-SAM3-p7/phase_b_prompt_ablation/` | Plan7-B: slope/curvature morphology ablation | Done |
| `Personal-Project/RS-SAM3-p7/phase_c_residual_boundary/` | Plan7-C3: SAM-HQ residual correction (no gain) | Done |
| `Personal-Project/RS-SAM3-p7/phase_d_multiscale_spatial/` | Plan7-D1: multi-scale TTA eval (no gain) | Done |
| `Personal-Project/RS-SAM3-p9/phase_a_lora_prompt/` | Plan9: LoRA + DSM edge/slope prompt (closed, -0.36pp vs F0'+L) | Done |
| `Personal-Project/RS-SAM3-p8/` | Plan8: cross-attn + attn bias (negative) | Done |
| `Personal-Project/RS-SAM3-p3/` `RS-SAM3-p3r/` `RS-SAM-p3b/` | Plan3 Route A/B: VPT adapter + UNet | Superseded |
| `Personal-Project/RS-SAM3-p1/` `RS-SAM3-p2/` `RS-SAM3-p4/` `RS-SAM3-p5/` `RS-SAM3/` | Plan1/2/4/5 | Superseded |
| `Reference-Project/SegEarth-OV-3-main/` | Zero-shot eval (Plan1) + SAM3 model loading | Read-only |
| `Reference-Project/sam3-main/` | Official SAM3 package (`pip install -e Reference-Project/sam3-main/`) | Read-only |
| `Reference-Project/MFNet/` `Reference-Project/SAM_RS/` | MFNet and SAM_RS papers (legacy) | Read-only |
| `docs/` | Paper notes for 10+ RS papers | Reference |
| `Plan/` | plan1-9.md research plans | Reference |
| `model_registry.py` | Authoritative registry of all checkpoints, metrics, lineage | **Single source of truth** |

Each experiment directory has its own isolated code (`model.py`, `train.py`, `eval.py`). Parameter signatures differ between versions — when loading old checkpoints, use the corresponding code.

## Datasets

**Data root**: `/root/autodl-tmp/dataset/`

| Dataset | Modality | Classes | Train/Test | Notes |
|---------|----------|---------|------------|-------|
| Vaihingen | NIRRG+DSM | 5 (excl. clutter) | 12/4 tiles | 9cm GSD, DSM pixel-aligned |
| Potsdam | RGBIR+DSM | 5 (excl. clutter) | 16/6 tiles | 5cm GSD |
| LoveDA | RGB | 7 | 2522/1669 | Largest training set |

### MFNet Standard Splits (DO NOT MODIFY)

```python
VAIHINGEN_TRAIN = ['1','3','23','26','7','11','13','28','17','32','34','37']
VAIHINGEN_TEST  = ['5','21','15','30']
POTSDAM_TRAIN   = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12',
                   '5_12','7_11','7_9','6_9','7_7','6_8','4_12','6_12']
POTSDAM_TEST    = ['4_10','5_11','2_11','3_10','6_11','7_12']
```

## Evaluation Protocol

**256×256 sliding window, stride=128, soft-logit accumulation (MANDATORY)**.

Every eval must output: `avg_oa`, `avg_miou`, `per_class_iou`, `per_class_recall`, `per_class_oa`.

Key rules:
- **Soft-logit accumulation**: accumulate logits across all patches before argmax (NOT per-patch argmax). Per-patch argmax loses ~0.4-0.8pp.
- Edge trim: `min(16, ph//4)` pixels from each patch edge.
- `per_class_recall = TP/(TP+FN)` — this matches MFNet paper's "per-class OA". Do NOT use `(TP+TN)/total` for comparison with MFNet.
- Multi-tile results use accumulated inter/union across all tiles (not tile-wise average).

**Training monitoring**: 512² or 1008² crop validation is for training monitoring only. It underestimates final mIoU by up to 27pp on Potsdam. Only 256² sliding window is final.

```bash
# Plan6 eval
python Personal-Project/RS-SAM3-p6/phase1_mm_adapter/eval_mfnet_protocol.py \
  --checkpoint <path> --dataset vaihingen

# Plan7 eval
python Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/eval.py --dataset vaihingen

# Plan7-C3 eval
python Personal-Project/RS-SAM3-p7/phase_c_residual_boundary/c3_full/eval.py \
  --checkpoint <path> --dataset vaihingen

# Multi-scale eval (D1/P2-D1)
python Personal-Project/RS-SAM3-p7/phase_d_multiscale_spatial/d1_multiscale_eval/eval_ms.py \
  --checkpoint <path> --dataset vaihingen --scales 1.0 0.75
```

## Key Commands

```bash
# Train Plan7-A (DSM edge/slope prompt)
cd Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt
python train_a.py --dataset vaihingen --dsm-attn-mode full --checkpoint-attn \
  --epochs 8 --batch 2 --init-from <Plan6_Phase1_best>

# Train Plan6 Phase1 (in-ViT MMAdapter)
cd Personal-Project/RS-SAM3-p6/phase1_mm_adapter
python train_phase1.py --dataset vaihingen --dsm-attn-mode full --checkpoint-attn

# Train Phase4 F0 (shared encoder + LoRA + SEFusion)
cd Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3
python train_f0.py --dataset vaihingen --epochs 12 --batch 2

# Train Phase4 F0' + LoRA (frozen + LoRA + 4xSEFusion)
cd Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline
python train_f0p_lora.py --dataset vaihingen --init-from <F0p_best> --epochs 8 --batch 2

# Train Plan9 (LoRA + DSM edge/slope input fusion)
cd Personal-Project/RS-SAM3-p7/phase_e_lora_prompt
python train_p9.py --dataset vaihingen --init-from <F0pL_or_F0_best> --epochs 8 --batch 2

# Install SAM3
pip install -e Reference-Project/sam3-main/

# List all registered models
python model_registry.py
```

All run outputs go to `/root/autodl-tmp/runs/`. Naming: `plan{N}_{phase}_{dataset}_{timestamp}/`.
Each run contains: `best_model.pt`, `history.json`, `config.json`, eval JSONs.

## Important Constraints

- **SAM3 fused ops don't support autograd**: `Reference-Project/SegEarth-OV-3-main/sam3/` has custom fused CUDA ops. For fine-tuning, use `Reference-Project/sam3-main/` pipeline or `strict=False` checkpoint loading.
- **Single-word class names only**: any prompt complexity beyond simple category words catastrophically degrades SAM3's text-vision alignment (Plan1 Phase 2).
- **Import path conflicts**: `Reference-Project/SegEarth-OV-3-main/sam3/` and `Reference-Project/sam3-main/sam3/` both export `sam3`. Scripts using `sam3-main` must remove SegEarth's sam3 from `sys.path`.
- **ViTDet block access**: `vision_backbone.trunk.blocks[i]`, not `vision_backbone.blocks[i]`. The `vision_backbone` is `Sam3DualViTDetNeck`.
- **Model registry is authoritative**: `model_registry.py` is the single source of truth for checkpoint paths, architectures, metrics, lineage. Consult it before loading any model. Update it after every experiment.
- **AGENTS.md** has subagent dispatch templates, coding style, and commit conventions.
- **CC工作模式.md** and **CC任务工作需求.md** define the master-session/subagent workflow for this project.
- **Lineage tracking**: every experiment must record `init_from`, `lineage_type` (`formal_ablation` or `continuation`), and `comparison_role`. Formal ablation must start from the same parent checkpoint.
- **Never modify** vendored reference projects in `Reference-Project/` unless the task explicitly targets them.

## Plan Closure Checklist

**Every plan must pass this checklist before being marked "已完成". No exceptions.**

When a plan (or a phase within a plan) reaches a conclusion:

```
□ plan{N}.md status updated: "状态: 已完成" with one-sentence conclusion
□ All checkpoints written to model_registry.py with: ckpt path, eval metrics, protocol, init_from, lineage_type, comparison_role, note
□ Experiment added to §Experiment Lineage above (with ✅/❌/➖ marker)
□ If negative result: added to §Failed Directions table above
□ CLAUDE.md "Current best" updated (if new record)
□ Run directory status decided:
    - Best checkpoint run dir: keep as-is
    - Other run dirs: tar.gz to /root/autodl-tmp/archives/, delete originals
    - Decision recorded in model_registry.py "archived_note" field
□ If formal ablation: the single variable changed and comparison baseline are unambiguous in both plan and registry
□ If failure: failure reason documented in plan{N}.md (not just "无效" — must state what was tried, what happened, and why it was abandoned)
```

**A plan without this closure record is considered "dangling"** — future work may inadvertently retry the same direction because the conclusion was never written down.
