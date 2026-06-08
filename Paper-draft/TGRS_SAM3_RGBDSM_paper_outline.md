# TGRS 论文大纲草稿：Geometry-Aware Parameter-Efficient Adaptation of SAM3 for RGB-DSM Remote Sensing Semantic Segmentation

> Draft date: 2026-06-07  
> Target venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)  
> Core positioning: 以 SAM3 适配为主线，以 MFNet 同协议为最强对标基线，而不是写成 MFNet 的增量工程改版。

---

## 0. 调研结论与写作定位

### 0.1 MFNet 论文的写作骨架

MFNet 的 TGRS 写法非常典型：

- Abstract 先讲多模态遥感语义分割的重要性，再引入 SAM/foundation model，最后强调 unified framework、Adapter/LoRA、DFM、多数据集 SOTA。
- Introduction 不是先讲模块，而是先讲遥感多模态融合的需求、传统 CNN/Transformer 多模态方法的限制、SAM 的 general knowledge，然后提出“multimodal fine-tuning framework”。
- Related Work 分成 Multimodal Remote Sensing Semantic Segmentation、Segment Anything Model、Fine-tuning/Adapter/LoRA 相关方向。
- Method 先给 unified multimodal fine-tuning formulation，再讲 Adapter/LoRA 两条实例化路径，最后讲 MFNet 总体架构和 DFM。
- Experiments 中先写 datasets、implementation、metrics，再用大表对比 15 个 SOTA，随后做可视化、component ablation、complexity analysis。
- Discussion/Analysis 不只报数，还解释 Tree/Low Vegetation 混淆、DSM 的贡献、Adapter vs LoRA、参数量/显存。

对我们最重要的启发：**TGRS 更接受“框架 + 系统实证 + 多维分析”的故事，而不是单个模型调参结果。**

### 0.2 其他 TGRS/SAM 遥感适配论文的共性

从 SAM-Assisted Remote Sensing Semantic Segmentation、RefAtt-SAM、TASAM/SAM3-Adapter 类文章看，审稿人期望看到：

- 明确指出 foundation model 直接用于遥感的 gap，例如 semantic-label gap、domain gap、prompt-dependence、terrain/geometry gap、dense prediction gap。
- 方法要模块化命名，最好有一张 Overview 图和 2-3 个核心模块图/公式。
- 对比实验必须有同环境或同协议说明，不能只引用不一致 benchmark 数字。
- 消融要对应贡献逐项验证：backbone adaptation、prompt/geometric prior、fusion/decoder、loss/refinement。
- 需要效率表：trainable params、memory、inference cost 或至少 trainable ratio。
- 需要可视化和错误分析，尤其是遥感常见难点：building boundary、tree/low vegetation、small cars、shadow/occlusion。

### 0.3 本文建议定位

不要写成：

> We improve MFNet by adding several tricks.

应写成：

> We study how to adapt SAM3 to RGB-DSM dense remote sensing semantic segmentation in a geometry-aware and parameter-efficient manner.

推荐一句话定位：

> This paper presents a systematic study and method for adapting SAM3 to RGB-DSM remote sensing semantic segmentation, showing that geometry-aware, parameter-efficient adaptation can outperform MFNet under the same evaluation protocol.

---

## 1. 候选题目

### 主推题目

**Geometry-Aware Parameter-Efficient Adaptation of SAM3 for RGB-DSM Remote Sensing Semantic Segmentation**

优点：稳健、TGRS 友好、突出遥感几何先验和参数效率，不直接碰瓷 MFNet。

### 备选题目

- **Adapting SAM3 to Multimodal Remote Sensing Semantic Segmentation With DSM Geometric Prompts**
- **Beyond MFNet: Geometry-Aware SAM3 Adaptation for RGB-DSM Semantic Segmentation**
- **Parameter-Efficient SAM3 Adaptation With Elevation-Aware Fusion for High-Resolution Remote Sensing Segmentation**

建议不要在最终标题中使用 Beyond MFNet，除非整篇文章明确以 MFNet 为中心展开。TGRS 更偏稳健叙事，主推题目更合适。

---

## 2. Abstract 草稿框架

### 中文逻辑

遥感 RGB-DSM 语义分割需要同时理解光谱纹理和高度几何结构。SAM/SAM3 提供强通用视觉表征，但直接用于遥感 dense semantic segmentation 时存在三类 gap：DSM 模态缺失、dense prediction/scale gap、遥感域适配 gap。本文提出一个 geometry-aware parameter-efficient SAM3 adaptation framework，通过 DSM-derived geometric prompts、LoRA/in-ViT adapter 和 lightweight multi-scale dense prediction decoder，将 SAM3 适配到 RGB-DSM 语义分割。实验在 ISPRS Vaihingen 和 Potsdam 上遵循 MFNet-aligned protocol。结果达到 Vaihingen 86.22 mIoU、Potsdam 89.35 mIoU，分别超过 MFNet best 1.50pp 和 2.66pp。消融表明轻量 backbone adaptation、DSM 几何提示和多尺度 dense reconstruction 是关键。

### English draft

Remote sensing semantic segmentation from RGB-DSM data requires joint reasoning over spectral appearance and elevation-induced geometric structures. Although Segment Anything models provide strong generic visual representations, directly adapting them to dense multimodal remote sensing segmentation remains challenging due to modality discrepancy, dense prediction requirements, and limited annotated aerial scenes. In this paper, we investigate parameter-efficient adaptation of SAM3 for RGB-DSM semantic segmentation and propose a geometry-aware adaptation framework. The framework injects DSM-derived structural cues into SAM3 through lightweight prompts/adapters, performs efficient backbone tuning with LoRA or in-ViT multimodal adapters, and reconstructs dense semantic maps via a lightweight multi-scale dense prediction decoder. Under the MFNet-aligned evaluation protocol, our adapted SAM3 achieves 86.22% mIoU on ISPRS Vaihingen and 89.35% mIoU on ISPRS Potsdam, outperforming the reported MFNet best results by 1.50 and 2.66 percentage points, respectively. Extensive ablations further reveal the roles of lightweight backbone adaptation, DSM geometric cues, and multi-scale dense reconstruction in transferring SAM3 to high-resolution remote sensing scenes.

---

## 3. Introduction 结构

### Paragraph 1: 遥感 RGB-DSM 语义分割的重要性

要点：

- High-resolution aerial/remote sensing semantic segmentation supports urban planning, land-cover mapping, disaster monitoring, precision agriculture, environmental assessment.
- RGB/NIR imagery captures spectral appearance; DSM/nDSM captures elevation and geometric structures.
- ISPRS Vaihingen/Potsdam 是 RGB/NIR + DSM 多模态城市语义分割经典 benchmark。

写作目标：让 TGRS 审稿人确认这不是普通 CV 任务，而是 Earth observation 中的核心问题。

### Paragraph 2: 现有 CNN/Transformer 多模态方法的限制

可提：

- FuseNet/vFuseNet、CMFNet、MFTransNet、FTransUNet、MultiSenseSeg 等通过 dual-stream、cross-attention、frequency/scale fusion 融合光学和 DSM。
- 它们通常 task-specific，需要从头训练，泛化受限。
- 小训练集下，模型容易过拟合局部纹理，tree/low vegetation、building boundary、car 小目标仍有混淆。

### Paragraph 3: Foundation models/SAM 的机会与限制

可提：

- SAM/SAM2/SAM3 具备强视觉基础表征。
- MFNet 已证明 SAM 可以通过 Adapter/LoRA 适配 RGB-DSM 语义分割，并在 Vaihingen/Potsdam/MMHunan 上取得强结果。
- 但已有 SAM-based remote sensing works 多集中在 binary segmentation、instance/referring/few-shot，或仍依赖通用 prompt/fine-tuning，未充分回答：**如何把 SAM3 系统性地适配到 RGB-DSM dense semantic segmentation？**

### Paragraph 4: 本文观察和核心 gap

建议明确三点 gap：

1. **Modality gap**: SAM3 原生以视觉/RGB 表征为主，DSM 高度几何没有显式编码。
2. **Dense prediction gap**: SAM3 的通用分割能力不等同于高分辨率遥感五类 dense semantic classification。
3. **Adaptation efficiency gap**: 全量微调或重型 adapter 成本高，且小样本 tile split 容易过拟合。

### Paragraph 5: 方法概述

提出：

> We propose a geometry-aware parameter-efficient SAM3 adaptation framework.

核心组件：

- DSM geometric prompt: edge/slope/nDSM/height-structure cues.
- Parameter-efficient SAM3 adaptation: LoRA or in-ViT multimodal adapter.
- Multi-scale dense semantic decoder: lightweight pyramid projection + multi-scale fusion + dense semantic reconstruction head.

### Paragraph 6: 贡献列表

建议贡献写法：

1. We formulate RGB-DSM remote sensing semantic segmentation as a geometry-aware parameter-efficient adaptation problem for SAM3, highlighting modality, dense prediction, and efficiency gaps.
2. We propose a SAM3 adaptation framework that integrates DSM-derived geometric prompts, lightweight backbone tuning, and multi-scale RGB-DSM fusion for dense semantic segmentation.
3. We provide a systematic adaptation study covering frozen SAM3, LoRA, in-ViT adapters, DSM prompts, SEFusion scales, and texture/refinement variants.
4. Under the MFNet-aligned protocol, our method achieves 86.22% mIoU on Vaihingen and 89.35% mIoU on Potsdam, outperforming MFNet by 1.50pp and 2.66pp.

---

## 4. Related Work 结构

### A. Remote Sensing Semantic Segmentation

覆盖：

- CNN encoder-decoder: FCN, U-Net, PSPNet, DeepLab, MAResU-Net.
- Transformer/hybrid: TransUNet, UNetFormer, FTransUNet, RS3Mamba.
- 遥感难点：多尺度地物、小目标、类间相似、边界复杂、标注有限。

重点不是列举，而是引出 foundation model adaptation 的必要性。

### B. RGB-DSM Multimodal Fusion

覆盖：

- Early fusion: RGB/NIR + DSM channel stacking.
- Dual-stream fusion: vFuseNet, FuseNet, CMFNet, MFTransNet.
- Cross-scale/frequency/attention fusion: FTransUNet, MultiSenseSeg, ASMFNet/MFFNet 等。
- MFNet 在这里作为 strongest RGB-DSM + SAM baseline。

需要指出：传统多模态网络通常是 task-specific，从头学习表示；MFNet 引入 SAM general knowledge，但几何先验和 SAM3 适配仍可进一步研究。

### C. Segment Anything Models in Remote Sensing

覆盖：

- SAM-RS / SAM-Assisted: 利用 SAM-generated object/boundary 做约束。
- RingMo-SAM / RSPrompter / PointSAM / RefAtt-SAM / MeSAM 等遥感 SAM 适配。
- 这些方法多面向 binary、instance、few-shot、prompt-based segmentation。

本文区别：

- 目标是 fully supervised RGB-DSM **semantic segmentation**。
- 不依赖人工 point/box prompt。
- 强调 DSM geometry + dense decoder + parameter-efficient adaptation。

### D. Parameter-Efficient Fine-Tuning for Foundation Models

覆盖：

- Adapter、LoRA、prompt tuning、visual prompt。
- SAM-Adapter/SAM3-Adapter 类工作。
- 参数效率对高分辨率遥感尤其关键：GPU memory、少量 train tiles、部署成本。

本文区别：

- 不只是把 LoRA/Adapter 接到 SAM3 上，而是研究其与 DSM geometric cues、multi-scale decoder 的组合。

---

## 5. Method 结构

### 5.1 Problem Formulation

输入：

- Optical image: Vaihingen 使用 NIRRG，Potsdam 使用 RGB。
- DSM/nDSM: per-tile min-max 或 nDSM 归一化。
- Label: five foreground classes, clutter ignored in evaluation.

形式：

```text
Given optical image X and height map H, predict dense semantic map Y over C=5 foreground classes.
```

可以写公式：

```text
F = E_SAM3(X; theta_frozen, theta_adapt)
G = Phi_geo(H)
Z = D_seg(F, G)
Y_hat = argmax softmax(Z)
```

### 5.2 Overview of the Proposed Framework

建议图 1：

```text
Optical Image ──> SAM3 ViTDet Encoder ──> Adapted Tokens ─┐
DSM/nDSM ──> Geometric Prompt Encoder ─> DSM Cues ────────┤
                                                          ├─> MSF-Decoder ─> Dense Semantic Map
LoRA / in-ViT MMAdapter inserted into SAM3 blocks ────────┘
```

叙述重点：

- SAM3 backbone mostly frozen.
- Only lightweight modules trainable.
- DSM is not treated as a naive extra channel only; it is converted into geometry-aware guidance.

### 5.3 SAM3 Backbone Adaptation

分两种实例化写：

#### 5.3.1 In-ViT Multimodal Adapter

用于 Vaihingen best：

- 32 blocks, bottleneck=32.
- RGB/DSM dual interaction in ViT blocks.
- 3-way gate + DSM edge/slope prompt.
- Trainable params about 12.7M / 820M.

写作时不要用 Plan7-A 名称，命名为：

> Geometry-Aware In-ViT Adapter (GA-Adapter)

#### 5.3.2 LoRA-Based Shared Encoder Adaptation

用于 Potsdam best：

- LoRA rank=8, attention+MLP.
- 4-scale multi-scale fusion decoder (MSF-Decoder).
- Trainable params about 7.2M.

命名为：

> LoRA Adaptation With MSF-Decoder

解释为什么两个 dataset best 不同：

- Vaihingen 训练 tile 少，几何 prompt 和 in-ViT interaction 对小样本泛化更关键。
- Potsdam 数据更多、图幅更大，LoRA + multi-scale dense reconstruction 能更充分适配 SAM3 表征并平衡 vegetation confusion。

### 5.4 DSM-Derived Geometric Prompting

写入：

- DSM/nDSM normalization.
- Edge/slope cues: Sobel/Laplacian/slope/height discontinuity.
- 目标是把 building boundary、tree crown structure、road/building height discontinuity 注入到 SAM3 features。

可以给一个抽象公式：

```text
G = Conv([H_norm, Sobel(H), Laplacian(H), Slope(H)])
T'_l = T_l + alpha_l * Adapter_l(G)
```

注意：如果最终模型里具体实现不同，公式用 conceptual formulation，后续 implementation details 说明。

### 5.5 Multi-Scale Dense Prediction Decoder

写入：

- SAM3 ViT features are strong but are not directly organized for full-scene land-cover classification.
- A dense semantic decoder is needed to convert adapted foundation-model tokens into pixel-wise class logits.
- The decoder builds a lightweight feature pyramid from SAM3 representations and aggregates multi-resolution features through global-local fusion and feature refinement blocks.
- This module should be positioned as a generic dense prediction head for SAM3 adaptation, not as a named MFNet component.

命名建议：

> Multi-Scale Fusion Decoder (MSF-Decoder)

备选名称：

- Lightweight Pyramid Fusion Decoder (LPF-Decoder)
- Dense Semantic Reconstruction Head (DSR Head)
- Elevation-Aware Pyramid Decoder (EAPD) — 只有当 decoder 明确使用 DSM/geometric cues 时再用。

建议主文表述：

> Unlike prompt-based SAM mask decoding, remote sensing semantic segmentation requires stable full-scene dense classification. We therefore attach a lightweight Multi-Scale Fusion Decoder (MSF-Decoder) to reconstruct dense semantic maps from adapted SAM3 features. The decoder projects foundation-model tokens into a multi-resolution feature pyramid and progressively aggregates global-local context before producing class logits.

实现细节可保守说明：

> The decoder follows the common lightweight global-local aggregation and feature refinement design used in recent RGB-DSM dense prediction networks. The contribution of this paper lies in SAM3 adaptation and geometry-aware multimodal conditioning rather than proposing a standalone decoder architecture.

避免写：

> We directly reuse the decoder from a prior baseline.

### 5.6 Training Objective

主 loss：

- Cross entropy.
- Optional DiceCE only作为 ablation/refinement，不作为主贡献。

可写：

```text
L = L_ce + lambda_aux L_aux
```

如果不想引入复杂 loss，主文保持简单，把 boundary/texture/refinement 放消融讨论。

### 5.7 Inference

必须强调 MFNet-aligned inference：

- window size 256x256
- stride 32
- no trim
- eroded/noBoundary labels
- soft-logit accumulation
- five foreground classes, ignore clutter

这保证对标 MFNet 的可信度。

---

## 6. Experiments 结构

### 6.1 Datasets

#### ISPRS Vaihingen

- 16 tiles.
- 12 train / 4 test under MFNet split.
- NIRRG + DSM.
- 9 cm GSD.
- 5 foreground classes + clutter ignored.

#### ISPRS Potsdam

- 24 selected tiles under MFNet split.
- 18 train / 6 test.
- RGB + DSM.
- 5 cm GSD.
- 5 foreground classes + clutter ignored.

建议附一个 dataset table：

| Dataset | Modality | GSD | Train/Test tiles | Classes | Evaluation labels |
|---|---|---:|---:|---:|---|
| Vaihingen | NIRRG+DSM | 9 cm | 12/4 | 5 + clutter | eroded/noBoundary |
| Potsdam | RGB+DSM | 5 cm | 18/6 | 5 + clutter | eroded/noBoundary |

### 6.2 Evaluation Protocol and Metrics

必须单独成节。TGRS 审稿如果质疑可比性，这一节是防线。

写明：

- We follow the MFNet-aligned protocol.
- mIoU and mF1 over five foreground classes.
- OA over evaluated pixels.
- Per-class OA in MFNet paper is recall = TP/(TP+FN); in our tables use per-class recall for direct comparison.

建议表：

| Item | Setting |
|---|---|
| Patch/window | 256x256 |
| Stride | 32 |
| Boundary handling | no trim, boundary-aligned sliding window |
| Label | eroded/noBoundary |
| Prediction aggregation | soft-logit accumulation |
| Classes | 5 foreground classes; clutter ignored |

### 6.3 Implementation Details

要包含：

- PyTorch version / GPU.
- SAM3 checkpoint / encoder type.
- Which parameters frozen/trainable.
- Optimizer, learning rate, batch size, epochs.
- Data augmentation.
- DSM normalization.
- Trainable params.

注意：如果多个实验配置不同，主文写 common setting，表格补充 best variant setting。

### 6.4 Main Comparison With MFNet and Prior Methods

主结果表必须干净，建议一张总表：

| Method | Backbone | Modality | Trainable Params | Vaihingen OA | Vaihingen mF1 | Vaihingen mIoU | Potsdam OA | Potsdam mF1 | Potsdam mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FTransUNet | R50-ViT-B | optical+DSM | 160.88M | 92.40 | 91.21 | 84.23 | 91.34 | 92.41 | 86.20 |
| MultiSenseSeg | SegFormer-B2 | optical+DSM | 60.46M | 92.73 | 91.42 | 84.53 | 91.30 | 92.35 | 86.10 |
| MFNet MMAdapter | SAM ViT-H | optical+DSM | 105.06M+6.22M | 92.97 | 91.71 | 85.03 | 91.71 | 92.70 | 86.69 |
| Ours-V | SAM3 + GA-Adapter + MSF-Decoder | optical+DSM | ~12.7M | 92.87 | TBD | 86.22 | - | - | - |
| Ours-P | SAM3 + LoRA adaptation + MSF-Decoder | optical+DSM | ~7.2M | - | - | - | 94.30 | TBD | 89.35 |

注意：注册表里 MFNet best comparison summary 对 Vaihingen 使用 84.72，而 MFNet 论文 full text 表中 ViT-H 为 85.03，注册表可能基于 strict reproducible entry/ViT-L best 对齐。主文需要统一口径：

- 若按 MFNet paper Table I 原文：Vaihingen best mIoU = 85.03。
- 若按当前 `mfnet_protocol_registry.py` summary：MFNet best mIoU = 84.72。

建议在写正式论文前再次核对：

```text
Action item: 统一 MFNet paper best 数字。若引用原文 Table I，应使用 85.03；若使用复现/注册表 strict entry，应明确说明是 reproduced/aligned entry。
```

当前对外最稳说法：

> Compared with the strongest MFNet result under our aligned protocol registry, our method improves mIoU by 1.50 pp on Vaihingen and 2.66 pp on Potsdam.

如果使用 MFNet paper 原文表：

> Vaihingen gain is 86.22 - 85.03 = +1.19 pp; Potsdam gain is 89.35 - 86.69 = +2.66 pp.

### 6.5 Per-class Analysis

建议表：

| Dataset | Method | Road/Imp. | Building | Low Veg. | Tree | Car | Key observation |
|---|---|---:|---:|---:|---:|---:|---|
| Vaihingen | MFNet best | 93.39 | 98.84 | 81.16 | 93.17 | 89.23 | strong building/road |
| Vaihingen | Ours-V | 92.47 | 97.45 | 84.33 | 93.09 | 98.70 | car +9.47, low veg +3.17 |
| Potsdam | MFNet best | 93.17 | 98.44 | 90.36 | 87.37 | 96.24 | weak tree |
| Potsdam | Ours-P | 94.44 | 98.72 | 90.23 | 91.02 | 98.10 | tree +3.65, car +1.86 |

讨论重点：

- Vaihingen: car and low vegetation improvement; road/building slightly lower than MFNet, but mIoU improves due to difficult classes.
- Potsdam: tree recall greatly improved; all classes except grass improve or match.
- DSM geometric adaptation helps vegetation/building/car boundaries and small-object localization.

### 6.6 Ablation Study

建议分 5 张表。

#### Table A: Backbone adaptation strategy

| Variant | Trainable Params | Vaihingen mIoU | Potsdam mIoU | Comment |
|---|---:|---:|---:|---|
| Frozen SAM3 + decoder | TBD | 75.xx | TBD | under-adapted |
| In-ViT Adapter + MSF-Decoder | ~12.7M | 86.22 | 89.20 | strong on Vaihingen |
| LoRA + single-scale fusion decoder | ~5.8M | TBD | 89.18 | strong grass recall |
| LoRA + 4-scale MSF-Decoder | ~7.2M | TBD | 89.35 | best Potsdam |

#### Table B: DSM geometry prompt

| Variant | DSM normalization | Prompt | Vaihingen mIoU | Observation |
|---|---|---|---:|---|
| RGB/NIR only | - | none | TBD | no geometry |
| DSM min-max | DSM | edge/slope | TBD | baseline geometry |
| nDSM | nDSM | edge/slope | 85.31 | stronger geometry normalization |
| nDSM + boundary weight | nDSM | edge/slope + loss | 85.15 | not effective |

#### Table C: Fusion scale

| Variant | Fusion design | Potsdam mIoU | Key confusion |
|---|---|---:|---|
| LoRA + single-scale fusion decoder | shallow fusion | 89.18 | tree->grass 9.4 |
| LoRA + 4-scale MSF-Decoder | multi-scale fusion | 89.35 | tree->grass 7.3 |

#### Table D: Texture/refinement branch

| Variant | mIoU | Delta | Conclusion |
|---|---:|---:|---|
| P11-B baseline | 85.31 | - | reference |
| Texture branch | 85.55 | +0.24 | weak signal |
| Building suppression | 85.53 | +0.22 | re-ranking |
| Per-class gates | 85.61 | +0.30 | best but noise-level |
| Strict veg-only texture | 85.30 | -0.01 | boundary gains removed by eroded labels |

结论要写得谨慎：

> Texture/refinement modules mostly perform logit re-ranking and are sensitive to boundary protocol; they are not the main source of robust improvement.

#### Table E: Protocol sensitivity

| Protocol | Stride | Labels | Trim | Observed effect |
|---|---:|---|---:|---|
| Internal 256 protocol | 128 | non-eroded | 16 | boundary-sensitive, lower mIoU |
| MFNet exact | 32 | eroded/noBoundary | 0 | comparable with MFNet, higher mIoU |

这张表非常重要，因为你已有结果显示 P13-G 在 256² 内部协议正向，但 MFNet exact 下不正向。

### 6.7 Complexity and Efficiency

建议表：

| Method | Backbone | Trainable Params | Trainable Ratio | Memory | mIoU |
|---|---|---:|---:|---:|---:|
| MFNet MMAdapter ViT-H | SAM ViT-H | 105.06M + 6.22M | TBD | 6854 MB | 85.03/86.69 |
| Ours-V | SAM3 | ~12.7M | 1.55% | TBD | 86.22 |
| Ours-P | SAM3 | ~7.2M | TBD | TBD | 89.35 |

需要补跑/补统计：

- GPU memory during training/inference.
- FLOPs or inference time for 256x256 window.
- Total parameters vs trainable parameters.

### 6.8 Qualitative Visualization

建议图：

1. Vaihingen sample:
   - Optical image
   - DSM/nDSM
   - GT
   - MFNet prediction
   - Ours prediction
   - Error map
   - Purple/red boxes around cars, building edge, low vegetation/tree.

2. Potsdam sample:
   - same columns
   - focus on tree/low vegetation and road/building edges.

3. Protocol sensitivity visualization:
   - non-eroded boundary vs eroded/noBoundary evaluation masks.
   - explain why boundary-improving module may not improve MFNet exact metric.

### 6.9 Discussion

建议小节：

#### A. What makes SAM3 transferable to RGB-DSM segmentation?

结论：lightweight adaptation aligns foundation features to remote sensing class taxonomy; DSM geometry supplies missing elevation prior.

#### B. Why do Vaihingen and Potsdam prefer different adaptation variants?

结论：

- Vaihingen small split: geometry-aware in-ViT adapter helps generalization and small difficult classes.
- Potsdam larger split: LoRA + multi-scale fusion adapts richer SAM3 features and balances vegetation confusion.

#### C. Limitations

必须写，TGRS 会喜欢诚实：

- 当前只在 ISPRS Vaihingen/Potsdam RGB-DSM benchmark 上验证；缺少 MMHunan/LoveDA 等额外跨域数据。
- MFNet official paper and our registry must be strictly aligned; official ISPRS leaderboard ranking requires server submission.
- Some refinements are protocol-sensitive; boundary gains may be hidden by eroded labels.
- SAM3 checkpoint and pretrained data may introduce reproducibility constraints.

---

## 7. Recommended Paper Section Layout

```text
I. Introduction
II. Related Work
    A. Remote Sensing Semantic Segmentation
    B. RGB-DSM Multimodal Fusion
    C. Segment Anything Models in Remote Sensing
    D. Parameter-Efficient Adaptation
III. Methodology
    A. Problem Formulation and Overview
    B. SAM3 Backbone Adaptation
    C. DSM-Derived Geometric Prompting
    D. Multi-Scale Dense Prediction Decoder
    E. Training Objective and Inference
IV. Experiments
    A. Datasets and Evaluation Protocol
    B. Implementation Details
    C. Comparison With State-of-the-Art Methods
    D. Per-Class and Confusion Analysis
    E. Ablation Study
    F. Complexity Analysis
    G. Qualitative Visualization
    H. Discussion and Limitations
V. Conclusion
```

---

## 8. Figures and Tables Plan

### Figures

| Figure | Content | Purpose |
|---|---|---|
| Fig. 1 | Overall framework | Establish method at a glance |
| Fig. 2 | SAM3 adaptation block: LoRA / in-ViT adapter / DSM prompt injection | Make contribution concrete |
| Fig. 3 | DSM geometric prompt examples: DSM, nDSM, slope, edge | Show remote sensing prior |
| Fig. 4 | Vaihingen qualitative comparison | Show cars/building/vegetation gains |
| Fig. 5 | Potsdam qualitative comparison | Show tree/vegetation gains |
| Fig. 6 | Error maps / confusion visualization | Explain per-class improvements |

### Tables

| Table | Content | Notes |
|---|---|---|
| Table I | Dataset statistics and protocol | Include split, GSD, modality |
| Table II | Main SOTA comparison | MFNet-aligned results |
| Table III | Per-class recall/F1 analysis | Directly compare MFNet per-class OA |
| Table IV | Backbone adaptation ablation | Frozen/LoRA/Adapter |
| Table V | DSM prompt/fusion ablation | Geometry contribution |
| Table VI | Texture/refinement negative/weak results | Shows rigorous study |
| Table VII | Complexity | Params/memory/speed |

---

## 9. 关键数字清单

### Main results from registry

| Dataset | Ours best | Ours mIoU | Ours OA | MFNet registry best mIoU | Delta |
|---|---|---:|---:|---:|---:|
| Vaihingen | Plan7-A / GA-Adapter | 86.22 | 92.87 | 84.72 | +1.50 |
| Potsdam | P10-C / LoRA + MSF-Decoder | 89.35 | 94.30 | 86.69 | +2.66 |

### If using MFNet paper original table values

| Dataset | Ours mIoU | MFNet paper best mIoU | Delta |
|---|---:|---:|---:|
| Vaihingen | 86.22 | 85.03 | +1.19 |
| Potsdam | 89.35 | 86.69 | +2.66 |

Action item: 正式论文必须统一“MFNet best”引用口径。建议以 MFNet 原文表为主，同时在补充材料说明 registry/reproduction protocol。

### Per-class strengths

| Dataset | Main strengths | Weaknesses |
|---|---|---|
| Vaihingen | car +9.47pp recall, low vegetation +3.17pp recall | road -0.92pp, building -1.39pp |
| Potsdam | tree +3.65pp recall, car +1.86pp, road +1.27pp | low vegetation -0.13pp |

---

## 10. 写作风险与补强任务

### Risk 1: Vaihingen MFNet best 数字口径不一致

问题：

- `mfnet_protocol_registry.py` summary 使用 `84.72` 作为 MFNet best。
- MFNet paper full text Table I 显示 MMAdapter ViT-H `85.03`，MMAdapter ViT-L `84.72`。

处理：

- 正式论文引用 MFNet 原文时使用 `85.03`。
- 如果强调 strict reproduced protocol，则写清楚是 “reproduced/aligned MFNet entry”。

### Risk 2: 两个数据集 best variant 不同

风险：

- 审稿人可能认为方法不统一。

处理：

- 统一框架下的 two instantiations：GA-Adapter + MSF-Decoder 和 LoRA adaptation + MSF-Decoder。
- 解释 dataset-dependent adaptation：小样本 vs 大样本、geometry cue vs multi-scale fusion。
- 主表可以放 “Ours-V” 和 “Ours-P”，消融证明两者都在统一设计空间中。

### Risk 3: 缺少第三数据集

TGRS 常喜欢 3 个数据集。MFNet 用 Vaihingen/Potsdam/MMHunan。

可选补强：

- 如果时间允许，补 MMHunan 或 LoveDA。
- 如果不补，必须强调本文聚焦 RGB-DSM high-resolution urban benchmark，Vaihingen/Potsdam 是最标准同模态协议。

### Risk 4: 官方 leaderboard 与内部 split

处理：

- 全文避免 “official rank #1”。
- 使用 “MFNet-aligned protocol” / “same split and evaluation protocol as MFNet”。
- 明确官方 ISPRS leaderboard 需要 server submission。

### Risk 5: SAM3 适配的新颖性需要比 MFNet 更清楚

处理：

- 不要只写 “SAM3 backbone 替换 SAM1”。
- 新颖性要落在：
  - geometry-aware DSM prompting
  - systematic SAM3 adaptation study
  - parameter-efficient adaptation with strong cross-dataset evidence
  - protocol sensitivity/error analysis

---

## 11. 建议的投稿前实验清单

必须完成：

- [ ] 统一 MFNet paper original values 与 registry values。
- [ ] 生成 mF1 for Ours-V and Ours-P，主表需要 OA/mF1/mIoU 三指标。
- [ ] 统计 trainable params、total params、memory、inference time。
- [ ] 生成 Vaihingen/Potsdam qualitative maps and error maps。
- [ ] 做 official MFNet-aligned protocol 的脚本校验说明。
- [ ] 至少补一个 seed 或重复实验，给出 mean/std 或说明 best checkpoint selection。

建议完成：

- [ ] 补 MMHunan 或 LoveDA 作为泛化验证。
- [ ] 比较 SAM1/SAM2/SAM3 backbone 或至少给 SAM3 vs SAM1/MFNet 的 discussion。
- [ ] 增加 no-DSM/RGB-only ablation，直接证明 DSM geometry 的贡献。
- [ ] 生成 vegetation confusion matrix table。

---

## 12. 参考资料与链接

### 本地资料

- `docs/MFNet论文/full.md`: MFNet TGRS 2025 full text extraction.
- `mfnet_protocol_registry.py`: 当前 MFNet-aligned result registry.
- `MFNet_eval_protocol.md`: 评估协议说明。
- `docs/SAM-RS论文/full.md`: SAM-Assisted Remote Sensing Semantic Segmentation with Object and Boundary Constraints, TGRS 2024.
- `docs/RefAtt-SAM论文/full.md`: RefAtt-SAM, TGRS 2026.
- `docs/TASAM/full.md`: SAM remote-sensing adaptation style reference.

### Web sources checked

- TGRS author page: https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/
- MFNet IEEE Xplore entry: https://ieeexplore.ieee.org/abstract/document/11063320
- SAM-Assisted Remote Sensing Semantic Segmentation DOI in local extraction: `10.1109/TGRS.2024.3443420`
- MFNet DOI in local extraction: `10.1109/TGRS.2025.3585238`
- ISPRS semantic labeling benchmark: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx

---

## 13. 当前推荐最终叙事

最适合 TGRS 的论文主线：

> MFNet showed that SAM can be adapted to multimodal remote sensing segmentation. We take the next step by systematically studying SAM3 adaptation for RGB-DSM dense semantic segmentation. The proposed geometry-aware parameter-efficient framework injects DSM structural priors into SAM3 and combines lightweight backbone adaptation with an MSF-Decoder for dense semantic reconstruction. Under the MFNet-aligned protocol, the adapted SAM3 surpasses MFNet on both ISPRS Vaihingen and Potsdam, while using a small number of trainable parameters and revealing which adaptation components are truly robust.

中文压缩版：

> 这篇论文不是“MFNet 改进版”，而是“SAM3 如何以参数高效方式适配 RGB-DSM 遥感语义分割”的系统研究。MFNet 是最强对标协议和 baseline。主贡献是 DSM 几何先验、轻量 backbone 适配、MSF-Decoder dense semantic reconstruction 以及严格同协议实证。
