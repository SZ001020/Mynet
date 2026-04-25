# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the SSRS (Semantic Segmentation for Remote Sensing) repo, containing PyTorch implementations of multiple published remote sensing works. The repo is organized as independent projects under one umbrella, each with its own training scripts, model definitions, and dependencies.

## Project Structure

- **`MFNet/`** — Multimodal fine-tuning framework using SAM as backbone (IEEE TGRS 2025). Key files:
  - `train.py` — Supervised training entry point
  - `train_uda_struct_v1.py` — Weak cross-domain UDA (unsupervised domain adaptation) training
  - `UNetFormer_MMSAM.py` — Model definition: SAM encoder + multimodal fusion + UNetFormer-style decoder
  - `utils.py` — Dataset config, data loading, losses (CE, BoundaryLoss, ObjectLoss), metrics, sliding window
  - `MedSAM/` — Deep dependency submodule with SAM model registry, LoRA/Adapter implementations

- **`SAM_RS/`** — SAM-assisted RS semantic segmentation with object and boundary constraints (IEEE TGRS 2024).
  - `train.py` — Training entry with multiple model options (UNetFormer, FTUNetFormer, ABCNet, CMTFNet)
  - `SAM_utils.py` — SAM preprocessing utilities
  - `model/CMTFNet/` — CMTFNet model implementation

- **`SegEarth-OV-3-main/`** — Open-vocabulary semantic segmentation using SAM 3 (zero-shot, no training).
  - `segearthov3_segmentor.py` — Main model (mmseg `BaseSegmentor` subclass wrapping SAM 3)
  - `eval.py` — MMSeg-based evaluation runner
  - `demo.py` — Quick inference demo
  - `custom_datasets.py` — MMSeg dataset registrations (PascalVOC, COCO, LoveDA, etc.)
  - `sam3/` — SAM 3 model code

- **`utils/`** — Shared utilities: `image_split.py`, `image_merge.py`, `draw_loss.py`

- **`docs/`** — Code structure guides and reference papers

- **`weights/`** — Pre-trained weights directory

## Configuration: Environment Variables

All training scripts are configured via **environment variables**, not config files. Key variables:

| Variable | Purpose | Default |
|---|---|---|
| `SSRS_DATASET` | Dataset name (`Vaihingen`, `Potsdam`, `Hunan`) | Vaihingen |
| `SSRS_DATA_ROOT` | Root data path | /root/SSRS/autodl-tmp/dataset |
| `SSRS_BATCH_SIZE` | Batch size | 10 |
| `SSRS_BASE_LR` | Learning rate | 0.01 |
| `SSRS_EPOCHS` | Training epochs | 50 |
| `SSRS_SEED` | Random seed | 42 |
| `SSRS_LOSS_MODE` | Loss composition (`SEG`, `SEG+BDY`, `SEG+OBJ`, `SEG+BDY+OBJ`) | SEG+BDY+OBJ |
| `SSRS_USE_STRUCTURE_LOSS` | Enable boundary/object losses | 1 |
| `SSRS_STRUCTURE_WARMUP_EPOCHS` | Warmup epochs for structure losses | 10 |
| `SSRS_LAMBDA_BDY` | Boundary loss weight | 0.1 |
| `SSRS_LAMBDA_OBJ` | Object loss weight | 1.0 |
| `SSRS_LOG_DIR` | Log and checkpoint directory | ./runs/week1_baseline |
| `SSRS_MFNET_SAM_ENCODER` | SAM encoder variant (`vit_b`, `vit_l`, `vit_h`) | vit_l |
| `SSRS_MFNET_SAM_CKPT` | SAM checkpoint path | weights/sam_vit_l_0b3195.pth |

## Running Training

**MFNet (supervised):**
```bash
cd MFNet
python train.py
```

**MFNet (UDA):**
```bash
cd MFNet
python train_uda_struct_v1.py
```

**SAM_RS:**
```bash
cd SAM_RS
python train.py
```

**SegEarth-OV-3 (evaluation):**
```bash
cd SegEarth-OV-3-main
python eval.py ./configs/cfg_DATASET.py
```

**SegEarth-OV-3 (demo):**
```bash
cd SegEarth-OV-3-main
python demo.py
```

## Key Architecture Patterns

### MFNet Model Flow
1. RGB (x) and DSM (y) inputs → shared SAM image encoder → `deepx`, `deepy` features
2. Each modality goes through FPN-style multi-scale projections (`fpn1x..fpn4x`, `fpn1y..fpn4y`)
3. Cross-modal fusion at each scale via `SEFusion` (squeeze-and-excitation channel attention)
4. UNetFormer-style decoder with Global-Local Attention blocks
5. Only `lora_` parameters in the encoder are trainable (PEFT with LoRA)

### Loss Design
- **Cross-entropy** — main segmentation loss
- **BoundaryLoss** — boundary-aware F1-based loss using boundary prior maps
- **ObjectLoss** — object-level consistency loss using instance maps
- Both auxiliary losses support confidence-gating (masking out low-confidence predictions)

### Sliding Window Inference
Large remote sensing images are processed via overlapping sliding windows, accumulating logits with averaging at overlap regions. Implemented in `utils.py` (`sliding_window`, `grouper`, `count_sliding_window`).

### Evaluation Metrics
- Per-class IoU, mean IoU (mIoU)
- Overall Accuracy (OA)
- mean F1 Score
- Kappa coefficient
- Confusion matrix

## Dataset Support

- **ISPRS Vaihingen** — 12 training / 4 testing tiles, NIRRG+DSM, 6 classes
- **ISPRS Potsdam** — 18 training / 6 testing tiles, RGBIR+DSM, 6 classes
- **MMHunan** — ~400 training / 50 testing tiles, RGB+DSM, 7 classes
- **SegEarth-OV-3**: Additional OpenEarthMap, LoveDA, iSAID, UAVid, WHU, Inria, DeepGlobe, etc.
