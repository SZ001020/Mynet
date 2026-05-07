# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the SSRS (Semantic Segmentation for Remote Sensing) repo, containing PyTorch implementations of multiple published remote sensing works. The repo is organized as independent research projects under one umbrella.

Active development is on the `segearthov3` branch, which focuses on SegEarth-OV-3 evaluation with SAM 3 for zero-shot open-vocabulary segmentation.

## Hardware & Environment

- GPU: RTX 5090 32GB (CUDA 12.8), PyTorch 2.7.0 with bf16 + FA3 support
- RAM: 754 GB
- Data disk: `/root/autodl-tmp` (100G) — store checkpoints and run results here
- Public disk: `/autodl-pub` (14T, 7.1T free) — for large model weights and dataset backups

## Research Context — READ THIS FIRST

The current branch has gone through two research plans. Understanding the negative results from Plan1 is essential before suggesting any approach.

### Plan1 (Multi-Class Paradigm) — `plan.md`

Concluded that **SAM3 should NOT be forced into a multi-class argmax model**:

| Phase | What | Result |
|-------|------|--------|
| Phase 1 | Zero-shot baseline + dual-head analysis | Semantic-Only head best for Vaihingen (65.8%), Dual-Head best for Potsdam (56.6%). Removing "clutter" class improves results. |
| Phase 2 | Prompt engineering (A-E groups) | **Single-word prompts = optimal.** Any complexity (RS terms, geometry descriptions, synonyms) catastrophically degrades performance (up to -50% mIoU). SAM3's text encoder is OOD for domain-specific prompts. |
| Phase 3 | Partial fine-tuning (FPN decoder) | **Fine-tuned worse than zero-shot** (Vaihingen: 65.8% → 56.6%, Potsdam: 55.6% → 21.0%). SAM3's fused operators (`perflib/fused.py`) don't support gradients. FPN decoder (797K params) too weak to learn text-vision alignment. |
| Phase 4 | DSM elevation injection | **DSM logit bias catastrophically fails** (Vaihingen: 43.3% → 15.4%). Simple post-hoc multiplication destroys SAM3's carefully trained logit distribution. |

### Plan2 (Per-Class Binary Paradigm) — `plan2.md`

Pivot to SAM3's native paradigm: **prompt → binary mask** per class, independent evaluation.

**Phase 1 — Completed (2026-04-30):** Per-class binary baseline in `RS-SAM3/`. Zero-cost threshold tuning yields +10.3% mIoU over baseline (0.450 → 0.554). Key finding: Plan1's multi-class IoU overestimated SAM3's real per-class capability by ~19%.

**Phase 2 — In progress:** SAM3 fine-tuning inspired by Medical-SAM3's checkpoint loading pattern. Three strategies: (A) Medical-SAM3 zero-shot transfer, (B) LoRA fine-tune on ISPRS, (C) full fine-tune on LoveDA. Implementation in `sam3_isprs/` using the official `sam3-main` training pipeline.

**Phase 3 — Future:** Train a dedicated "Remote-SAM3" checkpoint. DSM integration via hillshade visualization overlay (not model architecture changes).

### Plan3 (Three Routes for SAM3 Fine-Tuning) — `plan3.md`

Derived from Plan1's negative results and Plan2's per-class paradigm. Three complementary routes:

| Route | Approach | Trainable Params | Text Prompt | Risk |
|-------|----------|-----------------|-------------|------|
| A: Adapter+UNet | VPT adapters in ViT + UNet decoder | ~2M | No (vision-only) | Low |
| B: LoRA-Full-SAM3 | LoRA on ViT/Text/DETR/Mask decoders | 2M→15M | Yes | Medium |
| C: Two-stage | Route A validate → Route B complete | A:2M → B:15M | Eventually yes | Low→Medium |

**Key design principles:**
- Never modify SAM3's fused CUDA operators (keep frozen, no_grad)
- Inject trainable parameters only on standard PyTorch layers (Adapter on features, LoRA on nn.Linear)
- Maintain per-class binary paradigm from Plan2
- Route A (Adapter+UNet) is the fastest path to validate "fine-tuning can beat zero-shot"
- Route B (LoRA) preserves open-vocabulary capability
- Structure loss: edge-weighted BCE + weighted IoU for building/road boundary precision

## Project Components

### Active Development

#### `RS-SAM3/` — Plan2 Per-Class Binary Evaluation (active)
- **`eval_binary.py`** — Per-class binary evaluation runner. Outputs Dice/IoU/Precision/Recall per class to `runs/plan2_phase1_{timestamp}/`. Usage:
  ```bash
  cd /root/Mynet/RS-SAM3
  python eval_binary.py --dataset vaihingen [--max-samples N]
  ```
- **`tune_thresholds.py`** — Per-class top-K% threshold scanner. Sweeps K ∈ [3,5,8,10,12,15,18,20,25,30,35,40,50] on validation tiles, finds optimal per-class threshold.
- **`sam3_model.py`** — `SAM3Model` wrapper compatible with Medical-SAM3 checkpoint format
- **`dataset_rs.py`** — Per-class binary dataset loader (each tile × each class = one sample). Class definitions and text prompts in `TEXT_PROMPTS` dict.
- **`metrics.py`** — Dice, IoU, Precision, Recall computation

#### `fineNet/` — Plan1 Fine-Tuning Experiments (completed, reference)
- **`train.py`** — Phase 3 partial fine-tuning (frozen SAM3 backbone + trainable FPN decoder)
- **`train_dual_stream.py`** — Phase 4 dual-stream: RGB through frozen SAM3 + DSM through lightweight CNN encoder → UNet decoder
- **`train_phase3_unet.py`** — UNet decoder variant with ConvBlock + upsample layers
- **`train_phase4_dsm.py`** — Phase 4 dual-stream RGB+DSM fine-tuning (DSMEncoder + UNet decoder)
- **`phase4_approach_a.py`** — DSM logit bias post-processing (failed approach, kept for reference)
- **`phase4_dsm.py`** — DSM feature extraction and visualization
- **`phase3_visualize.py`** — Generate comparison visualizations (zero-shot vs fine-tuned predictions)
- **`finetune_model.py`** — FPN decoder model definition
- **`finetune_dataset.py`** — Training dataset loader (ISPRS tile-based)

#### `sam3_isprs/` — Official SAM3 Training on ISPRS (active)
Uses `sam3-main`'s training pipeline (not SegEarth's fused operators) for fine-tuning SAM3 on ISPRS:
- **`train_official.py`** — SAM3 fine-tuning via official `build_sam3_image_model(eval_mode=False)` with simplified detection loss on COCO-format data
- **`train_sam3_decoder.py`** — SAM3 feature extraction + UNet decoder (stronger decoder approach)
- **`train_simple.py`** — Simplified single-class training script
- **`convert_to_coco.py`** — Convert ISPRS tile annotations to COCO JSON format with per-class binary masks
- **`vaihingen/`, `potsdam/`, `combined/`** — COCO-format ISPRS data with annotations.json + tile images

#### `SegEarth-OV-3-main/` — Zero-Shot Evaluation (Plan1 Phases 1-2)
- **`eval.py`** — MMSeg-based evaluation runner. Usage:
  ```bash
  cd SegEarth-OV-3-main
  python eval.py ./configs/cfg_vaihingen.py [--out output_dir] [--show]
  ```
- **`demo.py`** — Quick single-image inference demo
- **`segearthov3_segmentor.py`** — `SegEarthOV3Segmentation` (mmseg `BaseSegmentor` subclass). Wraps SAM3 via `Sam3Processor`, performs per-category text-prompted inference with dual-head fusion (instance + semantic) and presence-guided filtering.
- **`configs/`** — One config file per dataset. Each inherits from `base_config.py`. Configs with `_noclutter` suffix exclude clutter class (5-class evaluation). Configs with `_prompt_b/c/d/e` suffixes correspond to Plan1 Phase 2 prompt experiments.
- **`cls_*.txt`** — Per-dataset class name lists (used as SAM3 text prompts)
- **`pamr.py`** — PAMR post-processing for mask refinement
- **`custom_datasets.py`** — MMSeg dataset registrations for 20+ RS datasets

### External SAM3 Codebases

#### `sam3-main/` — Official SAM 3 & SAM 3.1 (Meta)
Independent installable package. Contains the full model definition with SAM 3.1 multiplex tracking. This is **different** from `SegEarth-OV-3-main/sam3/` — newer, includes multiplex and SAM 3.1 support.
- Install: `pip install -e sam3-main/`
- Entry point: `sam3/model_builder.py`
- Training: `sam3/train/train.py` — official training pipeline (the correct path for fine-tuning, unlike the fused operators in `SegEarth-OV-3-main/sam3/`)
- Checkpoints auto-downloaded from HF: `facebook/sam3`, `facebook/sam3.1`

#### `Medical-SAM3/` — Reference Implementation (read-only)
Reference for checkpoint loading pattern used in Plan2 Phase 2. Demonstrates that SAM3 can be fine-tuned and weights reloaded with `strict=False`. Key insight: `build_sam3_image_model(checkpoint_path=None, load_from_HF=False)` + `model.load_state_dict(ckpt, strict=False)`.

#### `SAM3_LoRA-main/` — LoRA Fine-Tuning Implementation
Full LoRA implementation for SAM3 with training scripts and configs. Three training entry points:
- `train_sam3_lora.py` — Standard LoRA training
- `train_sam3_lora_native.py` — Native PyTorch training loop
- `train_sam3_lora_with_categories.py` — Category-aware training
- `lora_layers.py` — LoRA layer definitions
- `inference_lora.py` — Inference with LoRA weights
- Configs in `configs/` directory

#### `SAM3-UNet-main/` — SAM3UNet Paper Implementation
- `train.py` / `eval.py` — Training and evaluation
- `SAM3UNet.py` — Model combining SAM3 encoder with UNet decoder
- `sam3/` — Bundled SAM3 code subset

#### `mlx_sam3-main/` — Apple Silicon MLX Port
SAM3 ported to Apple's MLX framework. Not relevant for CUDA development.

### Legacy Projects (other branches: `week2`, `week3`, `master`)

#### `MFNet/` — Multimodal Fine-Tuning with SAM (IEEE TGRS 2025)
- **`train.py`** — Supervised training via environment variables
- **`UNetFormer_MMSAM.py`** — SAM encoder + multimodal FPN fusion + UNetFormer decoder with LoRA PEFT
- **`train_uda_struct_v1.py`** — Weak cross-domain UDA training
- Training config via env vars: `SSRS_DATASET`, `SSRS_DATA_ROOT`, `SSRS_BATCH_SIZE`, `SSRS_BASE_LR`, `SSRS_EPOCHS`, `SSRS_SEED`, `SSRS_LOSS_MODE`, `SSRS_LAMBDA_BDY`, `SSRS_LAMBDA_OBJ`

#### `SAM_RS/` — SAM-Assisted RS Segmentation (IEEE TGRS 2024)
- **`train.py`** — Supports 4 architectures: UNetFormer, FTUNetFormer, ABCNet, CMTFNet
- **`model/`** — Model implementations
- Uses same env var pattern as MFNet

## Key Architecture Patterns

### SegEarth-OV-3 Inference Pipeline
1. Input image + text class names → SAM3 model via `Sam3Processor`
2. Per-class text prompt → instance masks (Transformer decoder) + semantic logits (segmentation head)
3. **Instance aggregation** — consolidate sparse object predictions
4. **Dual-head fusion** — element-wise max of instance masks and semantic logits
5. **Presence filtering** — SAM3's presence score suppresses false positives from absent categories

### MFNet Model Flow
1. RGB (x) and DSM (y) → shared SAM image encoder → `deepx`, `deepy` features
2. FPN-style multi-scale projections per modality (`fpn1x..fpn4x`, `fpn1y..fpn4y`)
3. Cross-modal fusion at each scale via `SEFusion` (squeeze-and-excitation channel attention)
4. UNetFormer decoder with Global-Local Attention blocks
5. Only `lora_` parameters in the encoder are trainable (PEFT with LoRA)

### Per-Class Binary Paradigm (Plan2)
1. Image + single class text prompt → SAM3 → logits
2. Per-class thresholding (top-K% of activated pixels, K tuned per class) → binary mask
3. Evaluate Dice/IoU independently per class
4. No argmax competition between classes — each class stands on its own

## Experiment Results

All experiment outputs go to `/root/Mynet/autodl-tmp/runs/`. Naming convention:
- `phase1_baseline_YYYYMMDD_HHMMSS/` — Plan1 zero-shot experiments
- `phase2_prompt_YYYYMMDD_HHMMSS/` — Plan1 prompt engineering
- `phase3_partial_YYYYMMDD_HHMMSS/` — Plan1 fine-tuning runs
- `plan2_phase1_YYYYMMDD_HHMMSS/` — Plan2 per-class binary baselines
- `plan2_phase1b_YYYYMMDD_HHMMSS/` — Plan2 threshold tuning results
- `sam3_unet_combined_YYYYMMDD_HHMMSS/` — Plan3 Route A: SAM3+UNet fine-tuning runs

Each run directory contains `experiment.log`, CSV results, and per-tile visualization PNGs.

## Dataset Support (ISPRS focus for active work)

**Data root:** `/root/autodl-tmp/dataset/`

| Dataset | Modality | Classes | Train/Test | Path | Notes |
|---------|----------|---------|------------|------|-------|
| ISPRS Vaihingen | NIRRG+DSM | 6 (5 w/o clutter) | 12 train / 4 test | `/root/autodl-tmp/dataset/Vaihingen/` | DSM pixel-aligned with RGB |
| ISPRS Potsdam | RGBIR+DSM | 6 (5 w/o clutter) | 18 train / 6 test | `/root/autodl-tmp/dataset/Potsdam/` | DSM pixel-aligned with RGB |
| LoveDA | RGB | 7 | 2522 train / 1669 val | `/root/autodl-tmp/dataset/LoveDA/` | Largest available training set |

Other supported datasets (for zero-shot eval): OpenEarthMap, iSAID, UAVid, WHU, Inria, xBD, UDD5, VDD, CHN6-CUG, COCO Object/Stuff, PascalVOC, PascalContext, ADE20K, CityScapes, DeepGlobe Road, Massachusetts Road, SpaceNet Road, WBS-SI.

## Evaluation Metrics
- Per-class IoU, mean IoU (mIoU), Overall Accuracy (OA), mean Accuracy (mAcc)
- Plan2 adds: Per-class Dice, Precision, Recall (binary classification metrics)
- mean F1 Score, Kappa coefficient, Confusion matrix

## Standard Dataset Splits (MFNet Protocol)

**CRITICAL: All experiments use the MFNet standard train/test splits. Do NOT modify.**

These splits are aligned with the MFNet paper (IEEE TGRS 2025) for reproducible comparison.

```python
# MFNet Standard Splits — single source of truth
VAIHINGEN_TRAIN = ['1','3','23','26','7','11','13','28','17','32','34','37']  # 12 tiles
VAIHINGEN_TEST  = ['5','21','15','30']                                        # 4 tiles
POTSDAM_TRAIN   = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12','5_12','7_11','7_9','6_9','7_7','6_8','4_12','6_12']  # 16 tiles
POTSDAM_TEST    = ['4_10','5_11','2_11','3_10','6_11','7_12']                 # 6 tiles

# Tile name formats:
#   Vaihingen: top_mosaic_09cm_area{id}.tif / top_mosaic_09cm_area{id}.png
#   Potsdam:   top_potsdam_{id}_RGB.tif       / top_potsdam_{id}.png
```

**Usage by experiment:**
- Phase 1 (zero-shot eval): test tiles only
- Phase 2 (prompt engineering): test tiles only
- Phase 3 (fine-tuning): train tiles for training, test tiles for validation
- Phase 4 (DSM): same as Phase 3
- Plan2 Phase 1 (binary eval): test tiles only
- Plan3 Route A/B/C: train tiles for training, test tiles for validation

## Evaluation Protocol (CRITICAL — 2026-05-02)

**Final evaluation MUST use 256×256 sliding window with overlap averaging** (aligned with MFNet/ISPRS standard).

| Protocol | Window | Stride | Use | Reliability |
|----------|--------|--------|-----|-------------|
| 512² crop | 512² | — | Training monitoring only | **Underestimates** (Potsdam 47%→74%, -27pp gap) |
| 1008² sliding | 1008² | 672 | Intermediate reference | Overestimates (model trained at 1008²) |
| **256² sliding** | **256²** | **128** | **Final evaluation (MANDATORY)** | Matches MFNet paper protocol |

Implementation: `RS-SAM3-p3/eval_mfnet_protocol.py`
Usage:
```bash
cd RS-SAM3-p3
python eval_mfnet_protocol.py --model rgb --dataset vaihingen
python eval_mfnet_protocol.py --model dsm --dataset vaihingen
```

This protocol is documented in plan.md, plan2.md, and plan3.md.

### Code Version Management (CRITICAL — 2026-05-08)

Each experiment version has its own code directory. When loading old checkpoints, use the corresponding code — parameter signatures differ between versions.

**Model Registry:** `RS-SAM3-p4/model_registry.py` — maps every checkpoint → source code + loading params.
**Universal Eval:** `RS-SAM3-p4/eval_universal.py` — load any registered checkpoint, run 256² evaluation.

```bash
python eval_universal.py --list              # List all registered models
python eval_universal.py --name "VPT+MFNet"  # Evaluate specific model
python eval_universal.py                     # Evaluate all models
```

| Directory | Contents | Key Models |
|-----------|----------|------------|
| `RS-SAM3-p3/` | Route A: VPT Adapter + simple UNet | AdapterSAM3UNet |
| `RS-SAM3-p3r/` | Route A+DSM: VPT + UNetFormer + cross-attn DSM | AdapterSAM3UNetFormerDSM |
| `RS-SAM-p3b/` | Route B: LoRA ViT + UNetFormer + MFNet decoder | LoRASAM3UNetFormer, VPT_MFNetDecoder |
| `RS-SAM3-p4/` | Plan4: Full training + eval registry | SAM3FullTrain, model_registry.py |

### Evaluation Output Standard (ALL eval/validate must comply)

Every evaluate/validate function must output **four metrics**:

| Metric | JSON key | Description |
|--------|----------|-------------|
| Overall OA | `avg_oa` | Overall accuracy across all foreground classes |
| Overall mIoU | `avg_miou` | Mean IoU across all foreground classes |
| Per-class IoU | `per_class_iou` | Per-class IoU dict |
| Per-class OA | `per_class_oa` | Per-class OA dict: (TP+TN)/total |

Per-class OA formula (derived from existing inter/union, no extra accumulators needed):
```
pc_oa[c] = (inter[c] + total - union[c]) / total * 100
```
Derivation: TN = total - TP - FP - FN = total - union[c], so OA = (TP+TN)/total.

This enables direct comparison with MFNet paper's per-class OA (Table I reports per-class OA, not per-class IoU).

## Important Constraints

- **SAM3 fused operators don't support autograd**: `SegEarth-OV-3-main/sam3/` contains custom fused CUDA ops (`perflib/fused.py`, `vlcombiner.py`) that breaks gradient flow. For fine-tuning, use either `sam3-main/sam3/train/train.py` (official training pipeline) or Medical-SAM3's `strict=False` checkpoint loading pattern.
- **Single-word class names only**: Any prompt complexity beyond simple category words severely degrades SAM3's text-vision alignment. See Plan1 Phase 2 results.
- **Evaluate per-class, not via argmax**: Multi-class argmax overestimates SAM3's real class capability by ~19% mIoU due to competitive exclusion effects between classes.
- **Import path conflicts**: `SegEarth-OV-3-main/sam3/` and `sam3-main/sam3/` both export a `sam3` package. Scripts that use `sam3-main` must remove SegEarth's sam3 from `sys.path`:
  ```python
  sys.path.insert(0, '/root/Mynet/sam3-main')
  for p in list(sys.path):
      if p == os.path.join('/root/Mynet/SegEarth-OV-3-main', 'sam3'):
          sys.path.remove(p)
  ```
