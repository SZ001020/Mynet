# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作提供指导。

## 硬件与环境

- GPU: RTX 5090 32GB (CUDA 12.8), PyTorch 2.7.0, bf16 + FA3
- RAM: 754 GB
- 数据盘: `/root/autodl-tmp` (100G) — checkpoint、运行输出
- 公共盘: `/autodl-pub` (14T) — 模型权重、数据集备份

## 项目概述

SSRS 遥感语义分割研究仓库。当前分支: `segearthov3`。任务: 冻结 backbone 的 SAM3 在 ISPRS Vaihingen/Potsdam 上进行语义分割，融合 DSM（数字表面模型）。

**核心架构**: SAM3 ViTDet（冻结）+ in-ViT MMAdapter（每个 ViT block 内做 RGB/DSM 双流交互）+ MFNetDecoder。

**当前最佳（2026-05-19）**:

| 模型 | mIoU | OA | 可训练参数量 | 评估方式 |
|-------|------|-----|-----------|------|
| **Plan7-A: DSM edge/slope prompt** | **77.55%** | 88.20% | ~5M | 软 logit |
| Phase4 F0: shared + LoRA + SEFusion | 77.34% | 88.23% | ~7.2M | 软 logit |
| Plan6 Phase1: full DSM attention | 77.38% | 87.83% | ~5M | 软 logit |
| MFNet Frozen SAM1（参考基线） | 75.11% | 88.01% | LoRA | SAM1 |

**Phase 4 关键发现（2026-05-19）**: SAM3 ViTDet 上的 LoRA（rank=8）是最主要的影响因子（F0'→F0'+L 净贡献 +1.95pp），其重要性与 in-ViT MMAdapter 相当。Shared encoder + LoRA + 1×SEFusion（F0: 77.34%）与 Plan7-A（77.55%）差距仅 -0.21pp。LoRA + 简单架构 ≈ 冻结 + 复杂 adapter 架构。

## 实验谱系

每个 `→` 代表一次实验；✅/❌/➖ 分别标记正向/负向/噪声级结果。以 ❌ 结尾的方向已关闭，不应重试。

```
Plan6 Phase1 (76.57%) — 首个 in-ViT MMAdapter，无 prompt
  │
  ├─→ Plan7-A (+ Sobel/Laplacian prompt) → 77.14% ✅ +0.57pp  [当前最佳]
  ├─→ Plan7-B1 (slope only prompt) → 76.49% ❌ -0.65pp
  ├─→ Plan7-B3 (slope+edge+curvature) → 76.49% ❌ 无增益
  ├─→ Plan7-C3 (SAM-HQ residual correction) → 77.05% ➖ -0.09pp
  └─→ Plan7-D1 (multi-scale TTA) → 77.33% ➖ -0.22pp

Plan6 Phase3 (ablation) — 全部基于 Plan6 Phase1
  ├─→ A0: no adapter → 53.10%
  ├─→ A1: late fusion → 65.86% (+12.76pp)
  ├─→ A3: in-ViT MMAdapter → 72.88% (+6.75pp over A1) ← 关键消融
  ├─→ B1: softmax gate → +0.04pp ➖
  ├─→ C0: fixed windows → 噪声级
  └─→ D1: CE loss → 噪声级

Plan8 — SAM3 从头训练，seed=42
  │
  ├─→ CTRL (gate-only, Plan7-A architecture) → 74.77%
  ├─→ CA-A (+ DSM→RGB cross-attn, 4 global blocks) → 74.52% ❌ -0.26pp
  └─→ AB-A (+ DSM elevation attn bias) → 74.12% ❌ -0.65pp

Plan6 Phase4 — MFNet 严格对照，seed=42
  │
  ├─→ F0 (shared encoder + LoRA + 1×SEFusion + MFNetDecoder) → 77.34% ✅ [PHASE4 最佳]
  │   ├─→ F0 + AdamW + MultiStepLR 50ep → 76.57% ❌ -0.77pp
  │   └─→ F0 + SGD + MultiStepLR 50ep → 76.55% ❌ -0.79pp
  ├─→ F1 (in-ViT MMAdapter + LoRA) → 77.29% ➖ -0.11pp vs F0
  ├─→ F2 (in-ViT MMAdapter, frozen) → 75.51% ❌ -1.49pp vs F1
  └─→ F3 (in-ViT MMAdapter, frozen + prompt) → 76.52% (+0.65pp vs F2)

Plan6 Phase4 子链 — F0' 纯冻结基线
  │
  ├─→ F0' (SAM3 frozen + 4×SEFusion + MFNetDecoder) → 75.29% [对标 MFNet 75.11%]
  └─→ F0'+L (F0' + LoRA rank=8) → 77.24% ✅ +1.95pp ← LoRA 净贡献

Plan9 — LoRA + DSM edge/slope 融合
  │
  ├─→ Phase A (from F0'+L 77.24%): input-level 3ch DSM → 76.45% ❌ -0.36pp
  └─→ Phase B (from F0'+L): token-level PromptEncoder + gate (规划中)

Plan11 — 植被区分优化 (基于混淆矩阵诊断)
  │
  ├─→ P11-A (min-max + veg_boundary_weight, from Plan7-A) → 76.46% ❌ -0.68pp
  ├─→ P11-B (nDSM, from scratch, seed=42) → 76.72% ✅ +1.95pp vs Plan8-CTRL
  ├─→ P11-C (nDSM + veg_boundary_weight, from scratch) → 76.51% ❌ -0.21pp vs P11-B
  ├─→ P11-D (nDSM + augmentation, from scratch) → 76.56% ➖ -0.16pp vs P11-B
  └─→ P11-E (nDSM + 4-block adapter, from scratch) → 72.61% ❌ -4.11pp vs P11-B

早期 plan（已被超越）:
  Plan3: Adapter+decoder 以 175× 更少参数击败全量微调
  Plan4: 全量 ViT 训练导致退化；unfreeze 损害冻结 backbone
  Plan5: Boundary/object 辅助 loss 无增益
  Plan1-2: 零样本基线；per-class binary 范式高估
```

**谱系约定**:
- `formal_ablation`: 同一父 checkpoint，只改一个变量。可直接对比。
- `continuation`: 从最佳 ckpt 加载，改动多个变量。对比需谨慎。
- `from scratch`: 无继承权重。仅在同 seed 组内对比。
- 每个实验必须在其 `plan{N}.md` 和 `model_registry.py` 中记录 `init_from`、`lineage_type`、`comparison_role`。

### 已确认的失败方向（禁止重试）

| 方向 | 证据 | Δ vs 基线 | Plan |
|-----------|----------|---------------|------|
| Cross-attention RGB↔DSM in frozen ViT | CA-A 74.52 vs CTRL 74.77 | -0.26pp | Plan8 |
| DSM-derived attention bias | AB-A 74.12 vs CTRL 74.77 | -0.65pp | Plan8 |
| Full fine-tuning | 72.87 vs frozen 73.54 | 退化 | Plan4 |
| Unfreeze attention layers | 72.51 vs 76.57 | -4.06pp | Plan6 |
| Boundary/object auxiliary loss | — | 无增益 | Plan5 |
| SAM-HQ residual correction | 77.05 vs 77.14 | -0.09pp | Plan7-C3 |
| Multi-scale TTA | 77.33 vs 77.55 | -0.22pp | Plan7-D1 |
| Curvature/roughness prompt expansion | 76.49 vs 77.14 | -0.65pp | Plan7-B3 |
| Input-level DSM edge/slope channel concat | 76.45 vs 77.24 | -0.79pp | Plan9-A |
| Tree-grass boundary weighted loss | P11-A -0.68pp, P11-C -0.21pp | 持续负向 | Plan11 |
| Adapter 仅限全局注意力 block (4/32) | P11-E 72.61 vs P11-B 76.72 | -4.11pp | Plan11 |
| Photometric 数据增强 (ColorJitter+Blur) | P11-D 76.56 vs P11-B 76.72 | -0.16pp（噪声） | Plan11 |
| AdamW + MultiStepLR (F0, 50ep) | F0 76.57 vs Cosine 77.34 | -0.77pp | Plan6 F0 |
| SGD + MultiStepLR on small dataset (Vaihingen 12 tiles) | F0 76.55 vs Cosine 77.34 | -0.79pp | Plan6 F0 |

**提出任何新方向前，请先查此表和实验谱系。**

---

## 项目文件结构

### 目录分工

```
/root/Mynet/
├── Plan/                          # 研究计划文档（plan.md ~ plan10.md）
├── Personal-Project/              # 我们自己的实验代码（唯一可修改的代码区）
│   ├── RS-SAM3-p6/
│   │   ├── phase1_mm_adapter/     # Plan6 Phase1: in-ViT MMAdapter
│   │   ├── phase3_ablation/       # Plan6 Phase3: 消融实验
│   │   └── phase4_mfnet_ablation/ # Plan6 Phase4: MFNet 严格对照
│   │       ├── f0_mfnet_sam3/     #   F0: shared encoder + LoRA + SEFusion
│   │       └── f0p_frozen_baseline/ # F0': 纯冻结基线 + LoRA/MMLoRA
│   ├── RS-SAM3-p7/
│   │   ├── phase_a_dsm_prompt/    # Plan7-A: DSM edge/slope prompt
│   │   ├── phase_b_prompt_ablation/
│   │   ├── phase_c_residual_boundary/
│   │   └── phase_d_multiscale_spatial/
│   ├── RS-SAM3-p8/                # Plan8: cross-attn + attn bias（负向）
│   ├── RS-SAM3-p9/                # Plan9: LoRA + DSM edge/slope（负向）
│   ├── RS-SAM3-p10/               # Plan10: Potsdam 跨数据集验证
│   └── RS-SAM3-p3/ ... p5/        # Plan3/4/5（已被超越）
├── Reference-Project/             # 第三方参考代码（只读，禁止修改）
│   ├── SegEarth-OV-3-main/        # 零样本评估 + SAM3 模型加载
│   ├── sam3-main/                 # 官方 SAM3 包
│   └── MFNet/ SAM_RS/             # 论文参考
├── model_registry.py              # 模型注册表：所有 checkpoint、指标、谱系的权威来源
├── docs/                          # 论文笔记
└── CLAUDE.md                      # 本文件
```

### 新 Plan 执行时的文件结构规范

启动一个新 plan（如 plan N）时，按以下结构创建文件和目录:

#### 1. 代码文件 — 放在 `Personal-Project/RS-SAM3-p{N}/`

```
Personal-Project/RS-SAM3-p{N}/
├── phase_{X}_{描述}/              # 每个阶段一个子目录
│   ├── model.py                   # 模型定义
│   ├── train.py                   # 训练脚本
│   ├── eval.py                    # 评估脚本（必须实现 256² 滑动窗口）
│   └── dataset_adapter.py         # 数据加载（如需修改数据集逻辑）
```

**目录命名规范**:
- Plan 级别: `Personal-Project/RS-SAM3-p{N}/`（N 为 plan 编号）
- Phase 级别: `phase_{字母}_{简短英文描述}`，如 `phase_a_dsm_prompt/`、`phase_b_ablation/`
- 子实验放在 phase 目录下，如 `b1_slope_only/`、`b3_curvature/`

**创建命令**:
```bash
mkdir -p Personal-Project/RS-SAM3-p{N}/phase_{X}_{描述}
```

#### 2. 运行结果 — 放在 `/root/autodl-tmp/runs/`

```
/root/autodl-tmp/runs/
├── plan{N}_{phase}_{dataset}_{timestamp}/
│   ├── best_model.pt              # 训练中验证集最佳 checkpoint
│   ├── config.json                # 训练参数（由训练脚本自动保存）
│   ├── history.json               # epoch 级指标记录（loss、val_miou、val_oa）
│   ├── eval_256_{dataset}_{tag}.json  # 256² 滑动窗口评估结果
│   └── eval_256_{dataset}_{tag}_confusion.npy  # 混淆矩阵（numpy 格式）
```

**运行目录命名规范**: `plan{N}_{phase}_{dataset}_{timestamp}`
- 示例: `plan7_phase_a_dsm_prompt_vaihingen_20260510_225309`
- timestamp 由训练脚本在启动时自动生成: `datetime.now().strftime("%Y%m%d_%H%M%S")`

**每个运行目录至少包含**:
| 文件 | 说明 | 必需 |
|------|------|------|
| `best_model.pt` | 验证集 mIoU 最高的 checkpoint | 是 |
| `config.json` | 完整训练配置（参数、路径、超参） | 是 |
| `history.json` | 每个 epoch 的训练/验证指标 | 是 |
| `eval_256_*.json` | 256² 滑动窗口最终评估 | 是 |
| `eval_256_*_confusion.npy` | 混淆矩阵 `[num_classes, num_classes]` | 是 |

#### 3. Plan 文档 — 放在 `Plan/`

```
Plan/
└── plan{N}.md                    # 研究计划（实验设计、动机、假设）
```

#### 4. 归档

实验完成后:
- **最佳 checkpoint 运行目录**: 保留在原地
- **其他运行目录**: 打包为 `.tar.gz` 移至 `/root/autodl-tmp/archives/`，删除原始目录
- **在 `model_registry.py` 中记录归档决策**: 填写 `archived_note` 或 `deleted_note` 字段

---

## 评估协议与必须记录的数据

### 评估标准: 256² 滑动窗口（强制）

```
窗口大小: 256×256
步长: stride=128
边缘裁剪: min(16, patch_size // 4) 像素
累积方式: soft-logit 累积后 argmax（严禁 per-patch argmax）
跨 tile 聚合: 全局 confusion matrix 累积（非 tile-wise 平均）
```

**为什么 soft-logit 是强制的**: per-patch argmax 会损失 ~0.4-0.8pp mIoU。

### 必须保存的评估数据

每次 256² 滑动窗口评估完成后，**必须保存以下全部数据**:

#### JSON 文件 (`eval_256_{dataset}_{tag}.json`)

```json
{
  "avg_oa": 87.75,
  "avg_miou": 77.14,

  "per_class_iou": {
    "road": 77.61,
    "building": 87.87,
    "grass": 64.62,
    "tree": 78.16,
    "car": 77.44
  },

  "per_class_recall": {
    "road": 88.52,
    "building": 93.00,
    "grass": 79.70,
    "tree": 86.06,
    "car": 91.59
  },

  "per_class_oa": {
    "road": 92.68,
    "building": 95.94,
    "grass": 92.33,
    "tree": 93.88,
    "car": 99.72
  },

  "confusion_matrix": [[...], [...], ...],

  "checkpoint": "/path/to/checkpoint.pt",
  "checkpoint_epoch": 3,
  "protocol": "256² sliding window, stride=128, soft-logit accumulation",
  "dataset": "vaihingen"
}
```

#### 各字段说明

| 字段 | 含义 | 计算公式 |
|------|------|---------|
| `avg_oa` | 总体像素准确率 | `Σ TP_c / Σ (TP_c + FP_c)` 或 `Σ correct / Σ total` |
| `avg_miou` | 平均 IoU | `mean(per_class_iou)` |
| `per_class_iou` | 每类 IoU | `TP_c / (TP_c + FP_c + FN_c)` |
| `per_class_recall` | 每类召回率（即 MFNet 论文的 "per-class OA"） | `TP_c / (TP_c + FN_c)` |
| `per_class_oa` | 每类像素准确率（旧公式，不用于 MFNet 对比） | `(TP_c + TN_c) / total` |
| `confusion_matrix` | 混淆矩阵 | `[num_classes × num_classes]` 的原始计数（int64） |

**关键区分**:
- **`per_class_recall`** = TP/(TP+FN) = MFNet 论文的 "per-class OA"。与 MFNet 论文对比时用这个。
- **`per_class_oa`** = (TP+TN)/total。旧公式，car/grass 会被 TN 大幅拉高。**不可**与 MFNet 直接对比。

#### 混淆矩阵保存规范

混淆矩阵必须以 **原始整数计数（int64）** 保存，不做归一化:

```python
import numpy as np

# confusion 形状: [num_classes, num_classes]
# 行 = ground truth, 列 = prediction
np.save("eval_256_{dataset}_{tag}_confusion.npy", confusion_matrix.astype(np.int64))
```

此外，混淆矩阵也应内嵌在 eval JSON 中（`confusion_matrix` 字段），转为 Python list 便于直接查看。

从混淆矩阵可派生出所有指标:
```python
# 从混淆矩阵计算各项指标
cm = confusion_matrix  # [C, C], int64
tp = np.diag(cm)                        # 每类 TP
fp = cm.sum(axis=0) - tp                # 每类 FP
fn = cm.sum(axis=1) - tp                # 每类 FN

per_class_iou = tp / (tp + fp + fn)      # 每类 IoU
per_class_recall = tp / (tp + fn)        # 每类召回率 = MFNet "per-class OA"
avg_miou = np.mean(per_class_iou)        # 平均 IoU
avg_oa = tp.sum() / cm.sum()             # 总体像素准确率
```

### 训练过程监控 vs 最终评估

| 评估类型 | 用途 | 协议 |
|----------|------|------|
| 训练中 crop 验证（512²/1008²） | 选最佳 epoch、监控过拟合 | 仅参考，不可作为最终指标 |
| 256² 滑动窗口 | 最终指标、与其他模型/论文对比 | **唯一正式评估** |

**注意**: 512²/1008² crop 验证会大幅低估最终 mIoU（Potsdam 上可差 27pp）。只以 256² 滑动窗口结果作为最终指标。

### 评估脚本示例

```bash
# Plan6 Phase1 评估
python Personal-Project/RS-SAM3-p6/phase1_mm_adapter/eval_mfnet_protocol.py \
  --checkpoint <path> --dataset vaihingen

# Plan7 评估
python Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/eval.py \
  --checkpoint <path> --dataset vaihingen

# Multi-scale 评估
python Personal-Project/RS-SAM3-p7/phase_d_multiscale_spatial/d1_multiscale_eval/eval_ms.py \
  --checkpoint <path> --dataset vaihingen --scales 1.0 0.75
```

---

## 核心架构

### In-ViT MMAdapter（Plan6/7/8）

DSM token 与 RGB token 一起进入 SAM3 ViT 的每一层，共享冻结的自注意力机制:

```
RGB → ViT patch embed → RGB tokens (BHWC, 1024-dim)
DSM → CNN encoder → DSM tokens (相同空间网格)

每个 ViT block（共 32 blocks，global attn 在 [7,15,23,31]）:
  RGB attn = frozen_self_attention(RGB)
  DSM attn = frozen_self_attention(DSM)  ← "full" 模式复用相同权重
  gate = softmax(learnable_logits) → 3-way 融合 (RGB/DSM/prompt)
  output = gate[0]*RGB_adapter + gate[1]*DSM_adapter + gate[2]*prompt_adapter

ViT 之后: 多尺度特征（通过 backbone_fpn 从 blocks 8/16/24/32 输出）
  → Pyramid4Scale（4 尺度 FPN）
  → MFNetDecoder（3 GLA blocks + FeatureRefinementHead）→ 5 类 logits
```

Plan7 在此基础上增加第三路输入: 从 DSM 通过 Sobel/Laplacian 计算的 edge+slope prompt token，由单独的 PromptEncoder 编码后注入每层的 3-way softmax gate。

**关键实现细节**: ViTDet 通过 `vision_backbone.trunk.blocks` 访问（不是 `vision_backbone.blocks`）。`vision_backbone` 是 `Sam3DualViTDetNeck`，包装了 ViT `trunk`（32 blocks），每层由 `MMAdapterPromptBlock` 包装。

### MFNetDecoder

输入: 来自 Pyramid4Scale 的 4 尺度特征 [1/4, 1/8, 1/16, 1/32]。
架构: GLA（Global-Local Attention）blocks + window attention + relative position bias，WF 加权融合模块，FeatureRefinementHead（PA+CA 双注意力）。
输出: [B, 5, H/4, W/4] logits。

---

## 数据集

**数据根目录**: `/root/autodl-tmp/dataset/`

| 数据集 | 模态 | 类别数 | 训练/测试 | 备注 |
|---------|----------|---------|------------|-------|
| Vaihingen | NIRRG+DSM | 5（不含 clutter） | 12/4 tiles | 9cm GSD, DSM 像素对齐 |
| Potsdam | RGBIR+DSM | 5（不含 clutter） | 16/6 tiles | 5cm GSD |
| LoveDA | RGB | 7 | 2522/1669 | 最大训练集 |

### MFNet 标准划分（禁止修改）

```python
VAIHINGEN_TRAIN = ['1','3','23','26','7','11','13','28','17','32','34','37']
VAIHINGEN_TEST  = ['5','21','15','30']
POTSDAM_TRAIN   = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12',
                   '5_12','7_11','7_9','6_9','7_7','6_8','4_12','6_12']
POTSDAM_TEST    = ['4_10','5_11','2_11','3_10','6_11','7_12']
```

---

## 常用命令

```bash
# 训练 Plan7-A（DSM edge/slope prompt）
cd Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt
python train_a.py --dataset vaihingen --dsm-attn-mode full --checkpoint-attn \
  --epochs 8 --batch 2 --init-from <Plan6_Phase1_best>

# 训练 Plan6 Phase1（in-ViT MMAdapter）
cd Personal-Project/RS-SAM3-p6/phase1_mm_adapter
python train_phase1.py --dataset vaihingen --dsm-attn-mode full --checkpoint-attn

# 训练 Phase4 F0（shared encoder + LoRA + SEFusion）
cd Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3
python train_f0.py --dataset vaihingen --epochs 12 --batch 2

# 安装 SAM3
pip install -e Reference-Project/sam3-main/

# 列出所有注册模型
python model_registry.py
```

---

## 重要约束

- **SAM3 fused ops 不支持 autograd**: `Reference-Project/SegEarth-OV-3-main/sam3/` 包含自定义融合 CUDA ops。微调时使用 `Reference-Project/sam3-main/` 管线，或 `strict=False` 加载 checkpoint。
- **类名只能用单个词**: 任何超出简单类别词的 prompt 复杂度都会灾难性降低 SAM3 的文本-视觉对齐（Plan1 Phase 2）。
- **Import 路径冲突**: `Reference-Project/SegEarth-OV-3-main/sam3/` 和 `Reference-Project/sam3-main/sam3/` 都导出 `sam3`。使用 `sam3-main` 的脚本必须从 `sys.path` 中移除 SegEarth 的 sam3。
- **ViTDet block 访问**: `vision_backbone.trunk.blocks[i]`，**不是** `vision_backbone.blocks[i]`。
- **model_registry.py 是权威数据源**: checkpoint 路径、架构、指标、谱系的唯一真相来源。加载任何模型前先查它。每次实验结束后更新它。
- **禁止修改** `Reference-Project/` 下的第三方代码（除非任务明确要求）。
- **每个实验目录的代码是隔离的**（`model.py`、`train.py`、`eval.py`）。各版本参数签名可能不同——加载旧 checkpoint 时使用对应版本的代码。
- **CC工作模式.md** 和 **CC任务工作需求.md** 定义了主会话/subagent 工作流。

---

## Plan 关闭清单

**每个 plan 在标记为"已完成"前必须通过以下检查项。无例外。**

当一个 plan（或 plan 内的一个 phase）得出结论时:

```
□ plan{N}.md 状态已更新: "状态: 已完成"，附一句话结论
□ 所有 checkpoint 已写入 model_registry.py，包含: ckpt 路径、eval 指标、protocol、init_from、lineage_type、comparison_role、note
□ 实验已加入上方 §实验谱系（带 ✅/❌/➖ 标记）
□ 若为负向结果: 已加入上方 §已确认的失败方向表
□ 若创新记录: CLAUDE.md "当前最佳" 表已更新
□ 运行目录处置:
    - 最佳 checkpoint 运行目录: 保留
    - 其他运行目录: tar.gz 打包到 /root/autodl-tmp/archives/，删除原始目录
    - 归档决策记录在 model_registry.py 的 "archived_note"/"deleted_note" 字段
□ 若为 formal ablation: 单一变量变化和对比基线在 plan 和 registry 中表述清楚无歧义
□ 若为失败: 失败原因记录在 plan{N}.md 中（不能只写"无效"—必须写清尝试了什么、发生了什么、为什么放弃）
```

**无此关闭记录的 plan 视为"悬空"**——后续工作可能因结论未被记录而不经意间重试同一方向。
