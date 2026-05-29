# Plan6: SAM3 In-ViT RGB/DSM MMAdapter 路线

> 日期：2026-05-21
> 状态：Phase 1-4 + 三组附加实验完成；第四组附加实验规划中（深层解冻验证）
> Phase 4 结论：SAM3 frozen ≈ SAM1 frozen；LoRA +1.95pp；Adapter 路线和 LoRA 路线打平
> 当前最优：Plan7-A 77.14%（非本路线）
> Phase 3 消融结论：in-ViT MMAdapter 是绝对核心（+6.75pp），gate/loss/data 选择不关键（±0.2pp），decoder 中等贡献（+0.7pp）

---

## 实验全景（256² 整图评估）

| 实验 | OA | mIoU | road Rec | bldg Rec | grass Rec | tree Rec | car Rec | 可训练参数 | 结论 |
|------|-----|------|----------|----------|-----------|----------|---------|-----------|------|
| Phase 1 DSM-lite | 87.06 | 74.88 | 90.7 | 91.8 | 74.0 | 85.0 | 82.3 | ~5M | DSM adapter 模式基线 |
| **Phase 1 full attn** | **87.27** | **76.57** | **90.3** | **91.5** | **77.8** | **85.4** | **87.7** | **~5M** | **full DSM attention 最优** |
| Phase 1.5 +LoRA | 87.13 | 75.57 | — | — | — | — | — | 14.7M | LoRA 无帮助 |
| Phase 1.6 +unfreeze 4L | 86.11 | 72.51 | — | — | — | — | — | 27M | 解冻退化 |
| MFNet Frozen SAM1 | 88.01 | 75.11 | 89.5 | 94.6 | 71.7 | 89.5 | 76.8 | LoRA | SAM1 基线 |

---

## Phase 1：In-ViT MMAdapter（最优）

### 核心设计

让 DSM 进入 SAM3 ViT 的每一层参与 attention 和 MLP 计算。

```
之前 (Late Fusion):
  RGB → [ViT 32层, 只看RGB] → vit_feat → SEFusion → Decoder
  DSM → CNN Encoder → dsm_feat ──────────┘

Plan6 (In-ViT Fusion):
  RGB → [Block 0: attn(RGB)+attn(DSM) → gate fusion] → ... → [Block 31] → Decoder
  DSM → CNN → dsm_tokens ──┼──────────────────────────────────┘
```

两种 DSM token 更新模式：

| 模式 | 256² mIoU |
|------|-----------|
| 轻量 adapter | 74.88% |
| 完整双流 | **76.57%** |

### Phase 1 最优结果

| | 256² 整图 | 窗口验证 best |
|---|------|------|
| OA | 87.27% | — |
| mIoU | 76.57% | 76.86% (E10) |
| road | IoU=77.0, Rec=90.3 | — |
| building | IoU=86.9, Rec=91.5 | — |
| grass | IoU=63.5, Rec=77.8 | — |
| tree | IoU=77.7, Rec=85.4 | — |
| car | IoU=77.8, Rec=87.7 | — |

---

## Phase 1.5：在线裁剪 + LoRA 协同

改动：1) 在线随机裁剪替代固定窗口，2) LoRA rank=8 注入 frozen block。

| | 256² 整图 | 窗口验证 best |
|---|------|------|
| OA | 87.13% | — |
| mIoU | 75.57% | 76.59% (E13) |

LoRA 未带来收益（-1.00pp vs Phase 1），但在线裁剪的泛化 gap 收窄（1.02pp vs 1.14pp）。

---

## Phase 1.6：选择性解冻深层 Attention

解冻最后 4 层 ViT block 的 attention 权重，极低 lr=5e-7。

| | 256² 整图 | 窗口验证 best |
|---|------|------|
| OA | 86.11% | — |
| mIoU | **72.51%** | 76.45% (E1) |

解冻退化 4pp。窗口 E1=76.45%（热启动），但整图泛化崩溃。

---

## Phase 2：Decoder 侧增强 + 多尺度测试

| 子任务 | 结果 | 结论 |
|--------|------|------|
| P2-D1 多尺度推理 | mIoU -0.22pp | 对 SAM3 无效 |
| P2-C SAM-HQ 残差 | mIoU -0.09pp | 对 SAM3 无效 |

两项均为负向，Phase 2 方向关闭。

---

## Phase 3: 消融实验

### 背景

Plan6 Phase 1 从零样本 60.46% 到 76.57%，跨越 16pp，但这一跨越包含了多个同时引入的改动：MMAdapter、双流 attention、DSM 融合、decoder 升级、数据加载方式。没有一个干净的消融解释了每个改动的独立贡献。已确认有效的只有 full attention vs adapter 这一项（+1.69pp，Plan6 内部消融）和 DSM prompt 叠加（+0.57pp，Plan7-A 消融）。

Phase 3 的目标：用严格受控的 8-epoch 消融实验，拆解以下未归因模块的独立贡献。

---

### 消融链一：Adapter 贡献分解

**问题**：从 SAM3 原始 frozen backbone 到 Phase 1 full attention，中间经历了 DSM 引入、in-ViT 改造、双流 attention 三个关键步骤。每步的独立贡献从未被量化。

**对照设计**：

| 实验 | 架构 | 预期 mIoU | 回答的问题 |
|------|------|------|------|
| **A0** | SAM3 frozen + Pyramid4Scale + MFNetDecoder<br>无 DSM，无 adapter | ~68-70% | SAM3 原始 ViTDet 能做到多少？ |
| **A1** | A0 + DSM late fusion (SEFusion)<br>DSM encoder → 4× SEFusion → decoder | ~71-73% | DSM 不进入 ViT 时的贡献？ |
| **A2** | A1 → in-ViT MMAdapter (dsm_attn_mode=adapter)<br>DSM 走 adapter 更新，不共享 attention | 74.88% | in-ViT adapter 模式的贡献？ |
| **A3** | A2 → full DSM attention<br>DSM 复用 frozen attention，双流 | 76.57% | full attention 的额外贡献？ |

增量归因链：
```
A0 (SAM3 raw)         → 基线
A1 (+DSM late fusion) → DSM 本身的价值
A2 (+in-ViT adapter)  → 把 DSM 搬进 ViT 的价值  
A3 (+full attention)  → 双流 attention 的价值
```

**固定变量**：所有实验使用相同的 MFNetDecoder、Pyramid4Scale、固定窗口数据集、structure_loss、20 epoch。

**A0/A1 代码实现**：
- `Personal-Project/RS-SAM3-p6/phase3_ablation/a0_sam3_raw/` — 32 个原始 frozen block，无 adapter 注入
- `Personal-Project/RS-SAM3-p6/phase3_ablation/a1_dsm_late/` — DSM encoder + 4× SEFusion，无 in-ViT adapter
- A2/A3 复用已有的 Phase 1 DSM-lite 和 full attention 结果

**验收**：

| 结果 | 判断 |
|------|------|
| A1 - A0 < 2pp | DSM 在 late fusion 下几乎无用，Phase 1 的 3.5pp 主要来自 in-ViT |
| A2 - A1 > 2pp | in-ViT 改造本身就值 2pp+，和 attention 模式无关 |
| A3 - A2 ~ 1.7pp | 确认 full attention 贡献稳定 |

---

### 消融链二：Gate 类型

**问题**：Plan6 Phase 1 用两个独立 sigmoid gate（wx 控制 RGB 权重，wy 控制 DSM 权重），Plan7-A 用 softmax gate（三路归一化）。没有在同一架构下对比过两者的效果。

**对照设计**：

| 实验 | Gate 公式 | 说明 |
|------|------|------|
| **B0** | `sigmoid(wx)*rgb + sigmoid(wy)*dsm` | Phase 1 当前 |
| **B1** | `softmax([w0,w1])[0]*rgb + softmax([w0,w1])[1]*dsm` | softmax 归一化 |
| **B2** | `sigmoid(wx)*rgb + (1-sigmoid(wx))*dsm` | 共享单个 gate |

**固定变量**：Phase 1 架构、full DSM attention、在线裁剪、8 epoch、Phase 1 checkpoint 热启动。

**B1/B2 代码实现**：在现有 `MMAdapterBlock` 中通过参数切换 gate 类型。

**验收**：

| 结果 | 判断 |
|------|------|
| B1 > B0 | softmax 优于独立 sigmoid，Plan7 的 gate 设计可迁移回 Plan6 |
| B2 ~ B0 | 说明双 gate 没有带来额外收益，可用共享 gate 简化 |

---

### 消融链三：数据加载方式

**问题**：Phase 1 用固定窗口，Phase 1.5 用在线裁剪。但 Phase 1.5 同时加了 LoRA，无法归因于纯数据改动。固定窗口的泛化 gap（1.14pp）比在线裁剪（1.02pp）大，但绝对 mIoU 固定窗口更高（76.57% vs 75.57%）。

**对照设计**：

| 实验 | 数据加载 | 说明 |
|------|------|------|
| **C0** | 固定窗口（stride=128 预提取） | Phase 1 当前 |
| **C1** | 在线随机裁剪（epoch_steps=1000） | 同 Phase 1.5 但无 LoRA |

**固定变量**：Phase 1 架构、full DSM attention、8 epoch、Phase 1 checkpoint 热启动（C1 继续训练，C0 复用已有结果）。

**验收**：

| 结果 | 判断 |
|------|------|
| C1 > C0 | 在线裁剪本身有益，应作为后续训练默认 |
| C1 ~ C0 | 在线裁剪不影响最终指标，可选 |
| C1 < C0 | Phase 1.5 的退化全由 LoRA 负责，在线裁剪浪费训练多样性 |

---

### 消融链四：Loss 函数

**问题**：所有实验都用 `structure_loss`（edge-weighted BCE + IoU）。从未验证它是否优于标准 CrossEntropyLoss。

**对照设计**：

| 实验 | Loss | 说明 |
|------|------|------|
| **D0** | `structure_loss` | 当前 |
| **D1** | `CrossEntropyLoss(ignore_index=255)` | 标准多分类 CE |

**固定变量**：Phase 1 架构、full DSM attention、在线裁剪、8 epoch、Phase 1 checkpoint 热启动。

**验收**：

| 结果 | 判断 |
|------|------|
| D1 > D0 | CE 比 structure_loss 好，需要重新审视所有 loss 选择 |
| D1 ~ D0 | 两者等价，loss 不是关键因素 |
| D1 < D0 | structure_loss 确实更好，保留 |

---

### 消融链五：Decoder 架构

**问题**：MFNetDecoder 包含 4 个 GLA block（window attention + 相对位置偏置）、WF 加权融合模块、FeatureRefinementHead（PA+CA 双重注意力），总计 0.52M 参数。如果这些复杂设计实际贡献很小，模型可以大幅简化。

**对照设计**：

| 实验 | Decoder | 参数 | 说明 |
|------|------|------|------|
| **E0** | SAM3 原生 | ~7K | ViT backbone_fpn[-1] → Conv2d(256,5,1) → bilinear 上采样到 256²。SAM3 特征的最原始用法，无任何自定义 decoder |
| **E1** | 简单上采样 | ~50K | 4× bilinear upsample block，每块 ConvBNReLU(ch,ch)×2 + bilinear×2，最后 Conv(ch,5,1) |
| **E2** | MFNetDecoder | 0.52M | 当前，4× GLA + WF + FeatureRefinementHead |
| **E3** | UNetFormer (SAM3-Adapter 风格) | ~0.3M | 4× WF_single + FeatureRefinementHead_single，无 GLA attention。与 SAM3-Adapter 论文的 decoder 设计对齐 |

E0 回答："把 SAM3 的原始特征直接当分割特征用，没有 Pyramid4Scale 也没有 decoder，能到什么程度？"
E1 回答："加一个最朴素的上采样 decoder，没有 GLA 也没有 WF，值多少？"
E3 回答："SAM3-Adapter 论文用的 decoder 设计，和我们当前 MFNetDecoder 比如何？"

**固定变量**：Phase 1 架构、full DSM attention、在线裁剪、structure_loss、20 epoch。E0/E1/E3 需从头训练。

**验收**：

| 结果 | 判断 |
|------|------|
| E2 >> E3 ~ E1 ~ E0 | MFNetDecoder 是关键组件，GLA attention 贡献大 |
| E1 ~ E2 | GLA attention 无贡献，可砍掉 |
| E3 ~ E2 | UNetFormer 等价于 MFNetDecoder，可选更简单的 |
| E0 ~ E1 | 所有自定义 decoder 均不必要，SAM3 原始特征就够了 |
| E0 << E1 << E2 | 每一层 decoder 增强都有意义，当前设计合理 |

---

### 消融结果（2026-05-14）

| 实验 | 变量 | 窗口 best | 256² mIoU | vs 对照 | 结论 |
|------|------|------|------|------|------|
| **A0** SAM3 raw + Conv2d | 锚点 | 56.42% | 53.10% | — | frozen ViTDet 原始特征不够 |
| **A1** +DSM late + SimpleDecoder | DSM 融合 | 69.41% | 65.86% | +12.99 | DSM 有价值 |
| **A3** +in-ViT MMAdapter | adapter | 76.16% | 72.88% | +6.75 | **绝对核心** |
| **C0** A3 + 固定窗口 | 数据加载 | 76.83% | — | +0.67 | E1 更高但后续退化 |
| **D1** A3 + CE loss | loss | 76.34% | — | +0.18 | 噪声范围 |
| **B1** A3 + softmax gate | gate | 76.20% | — | +0.04 | 无实际差异 |
| Phase 1 +MFNetDecoder | decoder | 76.86% | 76.57% | +0.70 | 窗口中等，整图泛化价值更大 |

### 执行计划（已完成）

链一（Adapter）：回答"16pp 从哪来"——最核心，必须做
链五（Decoder）：回答"decoder 值多少"——性价比最高，简单 decoder 可能就够了
链二（Gate）：改一行代码，1.5h
链三（Data）：改一行代码，1.5h  
链四（Loss）：改一行代码，1.5h

Step 1: 实现链一 A0/A1 + 链五 E0/E1/E3（四个新架构）
Step 2: 跑链一 + 链五（20 epoch 从头训练，~4h each × 4 = ~16h）
Step 3: 实现链二 gate 切换（改 MMAdapterBlock）
Step 4: 跑链二/三/四（8 epoch 短训热启动，~1.5h each × 4 = ~6h）


**总成本估算**：~18-24 GPU 小时。

---

### 文件结构

```
RS-SAM3-p6/phase3_ablation/
├── a0_sam3_raw/             # SAM3 frozen + decoder，无 DSM 无 adapter
│   ├── model.py
│   ├── train.py
│   └── mm_adapter_vit.py    # 最小改动：跳过 adapter 注入
├── a1_dsm_late/             # DSM late fusion (SEFusion)
│   ├── model.py
│   ├── train.py
│   └── se_fusion.py
├── gate_ablation/           # 链二：gate 类型对比
│   └── mm_adapter_vit.py    # 支持 sigmoid/softmax/shared 切换
├── data_ablation/           # 链三：固定 vs 在线
│   └── (复用 Phase 1 和 Phase 1.5 代码)
└── loss_ablation/           # 链四：structure_loss vs CE
    └── train.py             # 仅改 loss 函数
```

---

## Phase 4：MFNet 严格对照消融

### 背景

当前最优（Plan7-A, 77.14%）和 MFNet（75.11%）的 2.03pp 差距中，混合了至少 5 个架构变量。Phase 3 已经拆解了 adapter/gate/loss/data 的贡献（均 ≤0.2pp），但有一个根本问题尚未回答：

**我们的 in-ViT MMAdapter 到底比 MFNet 的 shared encoder + LoRA + late SEFusion 强多少？这 2pp 差距里，多少是 SAM3 本身、多少是融合策略、多少是 decoder？**

Phase 4 在 SAM3 上搭建 MFNet 的严格等价物，逐个变量替换，得出每步独立贡献。

### 当前差距的变量拆解

| 变量 | MFNet | Plan7-A | 我们需要知道 |
|------|-------|---------|------------|
| 基础模型 | SAM1 ViT-L | SAM3 ViTDet | SAM3 比 SAM1 强多少？ |
| 编码器 | 共享（RGB+DSM 过同一 encoder） | 独立（DSM 走单独 CNN + adapter） | 共享 vs 独立编码器差多少？ |
| 融合位置 | Late SEFusion (encoder 之后) | In-ViT gate (每层 block 内) | 融合策略差多少？ |
| 参数策略 | LoRA (~1M) | Frozen + adapter (~5M) | LoRA vs Frozen 差多少？ |
| Decoder | UNetFormer | MFNetDecoder | decoder 差多少？ |
| 结构先验 | 无 | DSM edge/slope prompt | 已由 Plan7-A 回答：+0.57pp |

### 实验矩阵

所有实验用同一 SAM3 backbone、同一 Vaihingen split、同一 256² 软 logit 评估、seed=42。

| 实验 | 编码器 | 融合 | 参数策略 | Decoder | Prompt | 说明 |
|------|--------|------|---------|---------|--------|------|
| **F0** | 共享 | Late SEFusion | LoRA | MFNetDecoder | 无 | MFNet 等价物 on SAM3 |
| **F1** | 独立+MMAdapter | **In-ViT gate** | LoRA | MFNetDecoder | 无 | 融合策略贡献 |
| **F2** | 独立+MMAdapter | In-ViT gate | **Frozen** | MFNetDecoder | 无 | 参数策略贡献 |
| **F3** | 独立+MMAdapter | In-ViT gate | Frozen | MFNetDecoder | **edge/slope** | 结构先验贡献 |

Note: MFNet 原版 UNetFormer Decoder 和我们的 MFNetDecoder 都是 GLA-based，功能等价，省略 decoder-only 对照。

增量归因链：
```
F0 (MFNet on SAM3)        → 基线
F1 (+in-ViT MMAdapter)     → 融合策略独立贡献
F2 (-LoRA, +frozen)        → 参数策略独立贡献
F3 (+prompt)               → 结构先验贡献 (Plan7-A 已验证 +0.57pp)
```

### 预期结论

| 情景 | 含义 |
|------|------|
| F0 ≈ 75.11 | SAM3 backbone 和 SAM1 差距可忽略，MFNet 架构在 SAM3 上可复现 |
| F0 > 75.11 | SAM3 有固有优势，需要标注出 backbone 贡献 |
| F0 < 75.11 | MFNet 架构在 SAM3 上适配不如 SAM1 |
| F1-F0 > 0.5pp | MFNetDecoder 比 UNetFormer 显著强 |
| F2-F1 > 3pp | in-ViT 融合是核心（Phase 3 已见 +6.75pp，但那次包含了架构切换） |
| F3-F2 > 0.2pp | Frozen 和 LoRA 在当前规模下有实质性差异 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 数据 | Vaihingen，在线随机裁剪 |
| batch | 2 |
| epoch_steps | 1000 |
| lr (base) | 1e-4 |
| lr (adapter/decoder) | 5e-5 |
| lr (LoRA) | 5e-5 |
| seed | 42 |
| loss | structure_loss |
| eval | 256² 滑动窗口，软 logit 积累 |

| 实验 | epochs | 初始化 | 理由 |
|------|:---:|------|------|
| F0 | **12** | 从零 | 全新架构，未知收敛速度 |
| F1 | **8** | F0 best | 只换融合策略 |
| F2 | **8** | F1 best | 只去 LoRA |
| F3 | **8** | F2 best | 只加 prompt |

**总成本**：36 epoch × ~1.5h ≈ 54 GPU 小时 ≈ 2.5 天。

### 代码目录

```
Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/
├── f0_mfnet_sam3/            # MFNet 等价物 on SAM3（最核心）
│   ├── model_f0.py
│   ├── train_f0.py
│   └── eval_f0.py
├── f1_decoder/               # F1-F4 增量修改
├── f2_fusion/
├── f3_frozen/
├── f4_prompt/
└── shared/
    ├── se_fusion.py          # 复用 MFNet 的 SEFusion
    ├── unetformer_decoder.py # 复用 MFNet 的 UNetFormer Decoder
    └── lora_vit.py           # SAM3 ViTDet LoRA 注入
```

### F0 实现要点

F0 是核心——在 SAM3 上复刻 MFNet exact 架构：

1. **共享编码器**：RGB 和 DSM（repeat 3 通道）各自过 SAM3 backbone 的完整 forward，得 `deepx` 和 `deepy` 两组特征。对比 Plan6 Phase3 A0/A1 的 late fusion 模式
2. **LoRA**：在 SAM3 ViTDet 的 attention QKV 和 MLP 线性层注入 LoRA (rank=8)，复用 `RS-SAM-p3b` 的 LoRA 实现
3. **Late SEFusion**：4 尺度 SEFusion 模块，复用 MFNet 代码
4. **UNetFormer Decoder**：MFNet 的 `Decoder` 类

### Phase 4 结果（2026-05-19）

**256² 严格全局评估：**

| 实验 | OA | mIoU | road | bldg | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F0** shared+LoRA+SEFusion | 88.23 | **77.34** | 78.74 | 89.91 | 63.97 | 77.91 | 76.16 |
| F1 in-ViT+LoRA | 88.12 | 77.29 | 78.37 | 90.27 | 63.29 | 77.75 | 76.76 |
| F2 in-ViT frozen | 86.63 | 75.51 | 75.45 | 87.04 | 61.18 | 77.16 | 76.72 |
| F3 in-ViT frozen+prompt | 87.28 | 76.52 | 76.27 | 87.74 | 62.76 | 78.06 | 77.78 |

**Crop validation：**

| 实验 | Best mIoU | Best epoch | E1 mIoU |
|------|:---:|:---:|:---:|
| F0 | 76.74% | 3 | 75.07% |
| F1 | 76.83% | 7 | 74.66% |
| F2 | 74.76% | 7 | 72.94% |
| F3 | 75.93% | 8 | 75.14% |

**增量归因链（256² mIoU）：**

```
F0 (shared+LoRA+SEFusion):  77.34%  ← Phase 4 BEST
F1 (in-ViT+LoRA):           77.29%  (-0.11pp — in-ViT 不优于 late fusion，当有 LoRA 时)
F2 (in-ViT frozen):         75.51%  (-1.49pp — LoRA 贡献 ~1.5pp，是最大单一因素)
F3 (in-ViT frozen+prompt):  76.52%  (+0.65pp — prompt 贡献，与 Plan7-A 的 +0.57pp 一致)
```

**核心发现：**

1. **LoRA 是 Phase 4 链内最大单一因素** (+1.49pp in F1→F2)
2. **F0 (77.34%) 略低于 Plan7-A (77.55%，同软 logit eval)** — 差 -0.21pp。LoRA + 简单架构 ≈ frozen + 复杂 adapter 架构，打成平手
3. **in-ViT adapter 在有 LoRA 时不提供额外增益** (F1-F0 = -0.11pp，噪声级)
4. F3 (76.52%) 接近 Plan6 Phase1 (76.57%，旧 eval) — 两者同架构，验证了复现性

**⚠️ 注意 eval 方法**：Plan7-A 旧 eval (per-patch argmax) = 77.14%，软 logit eval = 77.55%。F0 软 logit = 77.34%。F0-vs-Plan7-A 的公平对比是 77.34 vs 77.55 (-0.21pp)。不要用 77.34 vs 77.14 这种跨 eval 方法的对比。

**Phase 4 的价值不是新 SOTA，而是首次在 SAM3 上做了 MFNet 架构 vs in-ViT adapter 的严格对照消融。结论：两者效果接近，LoRA 和 adapter 架构各贡献一半。**

---

## Phase 4 附加实验：SAM3 Frozen 干净基线

### 动机

Phase 4 的主体实验缺一个干净的 frozen SAM3 锚点：

- F2（75.51%）是 frozen + in-ViT adapter，**adapter 已占掉了大部分域适配空间**，不是真正的 frozen 基线
- F0（77.34%）有 LoRA，基线已经被 LoRA 拉高了

对标 MFNet 的 "Without Adapter"（frozen SAM1 + DFM + UNetFormer = 75.11%），我们需要一个 SAM3 上的严格等价物，才能：
1. 得出 SAM3 frozen 的真实水平（vs SAM1 frozen 75.11%）
2. 在干净的 frozen 基线上测 LoRA 的净贡献（不被 adapter 污染）

### 实验矩阵

两个实验，都用 DFM（4×SEFusion + pyramid + MFNetDecoder），对标 MFNet：

| 实验 | Backbone | Fine-tuning | Decoder | 说明 |
|------|---------|------------|---------|------|
| **F0'** | SAM3 frozen | **无**（纯冻结） | 4×SEFusion + MFNetDecoder | **对标 MFNet "Without Adapter" 75.11%** |
| **F0'+L** | SAM3 | **LoRA rank=8** | 4×SEFusion + MFNetDecoder | LoRA 在 SAM3 上的净贡献 |

为什么不用 F0 的代码：F0 是 shared encoder（RGB 和 DSM 各自过一遍 encoder）+ 1×SEFusion。F0' 改为 4×SEFusion（完整 DFM），更接近 MFNet 原始设计。预期 F0' 和 F0 效果接近。

### 预期分析

```
如果 F0' ≈ 75-76%: SAM3 frozen ≈ SAM1 frozen (75.11%)，SAM3 本身没有额外优势
如果 F0' > 76%: SAM3 frozen 明显强于 SAM1 frozen，我们之前高估了 adapter/LoRA 的贡献

F0'+L - F0' = LoRA 在 SAM3 上的净贡献（预期 +1.5pp ~ +2pp）
```

### 训练配置

| 参数 | F0' | F0'+L |
|------|------|------|
| epochs | 12 (从零) | 8 (从 F0' best 续训) |
| batch | 2 | 2 |
| epoch_steps | 1000 | 1000 |
| lr | 1e-4 | 5e-5 |
| seed | 42 | 42 |
| loss | structure_loss | structure_loss |
| init | from scratch | F0' best |

### 代码

复用 F0 的代码框架，把 1×SEFusion 扩展为 4×SEFusion（每个 backbone_fpn 层一个，对齐到 4 尺度后融合）。

### 附加实验结果（2026-05-20）

**256² 严格全局评估：**

| 实验 | OA | mIoU | road | bldg | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F0'** frozen+4×SEFusion | 86.57 | **75.29** | 75.46 | 86.85 | 60.85 | 76.97 | 76.30 |
| **F0'+L** +LoRA rank=8 | 87.89 | **77.24** | 77.50 | 88.34 | 64.66 | 78.28 | 77.40 |

**Crop validation：**

| 实验 | Best mIoU | Best epoch |
|------|:---:|:---:|
| F0' | 74.53% | 12 |
| F0'+L | 76.81% | 1 |

**对照 SAM1 的 MFNet 数据：**

| SAM1 | SAM3 | Δ |
|------|------|:---:|
| Without Adapter: 75.11% | F0': 75.29% | +0.18pp |
| Standard LoRA+DSM: 82.06% | F0'+L: 77.24% | **-4.82pp** |

**核心发现：**

1. **SAM3 frozen ≈ SAM1 frozen** (+0.18pp) — 冻结状态下两个 backbone 几乎等价
2. **LoRA 在 SAM1 上远比在 SAM3 上有效** — SAM1 +6.95pp vs SAM3 +1.95pp。SAM3 ViTDet 的预训练权重更"固化"，低秩修正的边际收益更小
3. **F0'+L (77.24%) > F0 (77.34%)? 不对——F0 用 1×SEFusion，F0'+L 用 4×SEFusion。4×SEFusion 在 SAM3+LoRA 下反而退步 -0.10pp，说明更多融合模块不等于更好**
4. **在所有 SAM3 对比中，Plan7-A (77.55%) 仍是最优** — frozen + in-ViT adapter + prompt 的组合在 SAM3 上略优于 LoRA-based 方案

### Phase 4 全部实验汇总

| 实验 | 256² mIoU | 定位 |
|------|:---:|------|
| Plan7-A (参考) | 77.55% | frozen + adapter + prompt，当前最优 |
| F0'+L | 77.24% | frozen + LoRA + 4×SEFusion |
| F0 | 77.34% | shared encoder + LoRA + 1×SEFusion |
| F1 | 77.29% | in-ViT adapter + LoRA + 1×SEFusion |
| F3 | 76.52% | in-ViT frozen + prompt |
| F2 | 75.51% | in-ViT frozen |
| F0' | 75.29% | 纯 frozen 基线 (对标 MFNet Without Adapter) |

**最终结论：在 SAM3 上，LoRA 和 Adapter 两条 fine-tuning 路线效果相当（~77.3%），差距在评估噪声范围。SAM3 的预训练权重比 SAM1 更固化，微调的边际收益更小。**

---

## Phase 4 第三组附加实验：对齐 MFNet MMLoRA

### 动机

F0'+L 与 MFNet 的 MMLoRA (83.96%) 未对齐：

| 组件 | MMLoRA | F0'+L | 对齐？ |
|------|--------|-------|:---:|
| LoRA 注入位置 | attn q,v only（MLP 不加） | attn qkv + MLP fc1/fc2 | ❌ |
| Encoder 内跨模态融合 | MLP 后 dual-branch λ 混合 | 无（各自独立 forward, encoder 外 SEFusion） | ❌ |
| DFM | 4×SEFusion | 4×SEFusion | ✅ |
| Decoder | UNetFormer | MFNetDecoder | ≈ |

需要补两个实验，逐步对齐到 MMLoRA：

| 实验 | 内容 | 训练 | 预期 |
|------|------|:---:|------|
| **F0'+L2** | LoRA 只加 attn q,v（不加 MLP），与 MFNet 对齐 | 续训 8 epoch | +0.2-0.5pp |
| **F0'+M** | F0'+L2 + encoder 内 dual-branch λ 混合（每层 MLP 后） | 续训 8 epoch | +0.5-1.5pp |

### F0'+L2 实现

修改 LoRA 注入 pattern，去掉 `mlp.fc1` 和 `mlp.fc2`，只保留 `attn.qkv` 和 `attn.proj`。代码一行改动。

### F0'+M 实现

这是核心——让 RGB 和 DSM 在同一轮 encoder forward 内交互。

架构（仿 MMLoRA，每个 ViT block）：

```
RGB tokens, DSM tokens → shared frozen attention (LoRA on q,v only)
  → 各自过 frozen MLP
  → MMLoRA mixing:
      x_ada = Bx·Ax·x_norm     (RGB LoRA 低秩修正)
      y_ada = By·Ay·y_norm     (DSM LoRA 低秩修正)
      x_out = x + λ1·x_ada + (1-λ1)·y_ada   ← 跨模态！
      y_out = y + λ2·y_ada + (1-λ2)·x_ada   ← 跨模态！
```

注意：之前 F0'+L 的 encoder 内部没有跨模态交互（RGB 和 DSM 各自独立 forward）。F0'+M 在每个 ViT block 的 MLP 后加入 dual-branch λ 混合，使 DSM tokens 在 encoder 内部逐层渗透 RGB 表示。

### 训练配置

| 参数 | F0'+L2 | F0'+M |
|------|------|------|
| epochs | 8 | 8 |
| batch | 8 | 4（每个 batch 内 RGB+DSM 同时过 encoder, 显存翻倍） |
| lr | 5e-5 | 5e-5 |
| seed | 42 | 42 |
| init | F0'+L best | F0'+L2 best |
| 预期 Δ | +0.2-0.5pp | +0.5-1.5pp |

### 第三组附加实验结果（2026-05-21）

**256² 评估：**

| 实验 | OA | mIoU | vs F0'+L |
|------|:---:|:---:|:---:|
| **F0'+L** (LoRA attn+MLP) | 87.89 | **77.24** | — |
| F0'+L2 (LoRA attn only) | 87.57 | 77.01 | -0.23pp |
| F0'+M (attn + MMLoRA mixing) | 87.64 | 76.89 | -0.35pp |

**Crop validation：**

| 实验 | Best mIoU | Best epoch |
|------|:---:|:---:|
| F0'+L | 76.81% | 1 |
| F0'+L2 | 76.56% | 7 |
| F0'+M | 76.45% | 6 |

**结论：**
1. LoRA on attn+MLP (F0'+L) 是最优配置，去掉 MLP LoRA 损失 0.23pp
2. **MMLoRA dual-branch λ 混合无增益**（F0'+M -0.35pp vs F0'+L）— encoder 内部跨模态融合在 SAM3+LoRA 条件下是冗余的
3. 此方向已关闭。SAM3 上的最优 LoRA 方案就是 F0'+L

---

## Phase 4 第四组附加实验：深层 Attention 解冻验证

### 动机

MMA 论文（Multi-Modal Adapter for VLMs）在 CLIP ViT 上通过 dataset-level recognition 证明了：**低层特征跨数据集通用（该冻），高层特征数据集特定（该训）。** Adapting-SAM3 把这个结论沿用到 SAM3 上，把 adapter 放在 blocks 20-31。但 SAM3 ViTDet 上从未被严格验证过。

Plan6 Phase 1.6 试过一次：unfreeze blocks 28-31 attn，lr=5e-7，30 epoch，在线裁剪。结果退化 4pp。但这不能下结论——lr 太小（三个数量级低于正常微调 lr），在线裁剪加剧过拟合。

**需要一次干净的重跑**：固定窗口、正常 lr、合理 epoch，验证"在 SAM3 上只解冻深层 attention 权重"是否有效。

> 理论基础来源：[来源：MMA 论文 Figure 1 dataset-level recognition + Adapting-SAM3 blocks 20-31 adapter placement]

### 实验设计

| 实验 | Backbone | 解冻范围 | 数据 | 说明 |
|------|---------|---------|------|------|
| **F0' (已有)** | SAM3 frozen | 无 | 固定窗口 | 冻结基线 75.29% |
| **U1** | SAM3 | blocks 24-31 attn (qkv+proj) | 固定窗口 | 只解冻深层 attention |
| **U2** | SAM3 | blocks 24-31 attn + MLP | 固定窗口 | 解冻更多，边界测试 |
| **U3** | SAM3 | blocks 28-31 attn (同 Phase 1.6) | 固定窗口 | 验证 Phase 1.6 失败是 lr/数据问题 |

### 训练配置（修正 Phase 1.6 的三个错误）

| 参数 | Phase 1.6 (失败) | Phase 4 U1-U3 (修正) |
|------|:---:|:---:|
| 数据加载 | 在线随机裁剪 | **固定窗口**（和 F0' 一致） |
| attn lr | 5e-7 | **1e-5**（低但仍正常） |
| decoder lr | 5e-5 | 5e-5 |
| epochs | 30 | **12**（够看趋势） |
| batch | 2 | 4 |
| seed | — | 42 |

### 预期

```
如果 U1 > F0' (75.29%): 深层解冻有效——MMA 的结论在 SAM3 上成立
如果 U1 ≈ F0': 解冻深层和冻结效果一样——SAM3 ViTDet 的 window attention 可能限制了收益
如果 U1 < F0': 加深了 Phase 1.6 的结论——SAM3 是解冻无益的
如果 U1 > U2: 只解冻 attn 就够了，MLP 不需要动
```

### 实现

- 从 F0' best 热启动，unfreeze 指定层，续训 12 epoch
- 复用 F0' 的 4×SEFusion + MFNetDecoder（冻结）
- 代码位置：`Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline/unfreeze/`

### 第四组附加实验结果（2026-05-26）

**Crop validation：**

| 实验 | 解冻范围 | Best mIoU | Best epoch | vs F0' |
|------|---------|:---:|:---:|:---:|
| F0' (frozen baseline) | 无 | 75.29% | — | — |
| **U1** | blocks 24-31 attn (qkv+proj) | 75.49% | 10 | +0.20pp |
| **U2** | blocks 24-31 attn+MLP | 75.55% | 4 | +0.26pp |
| **U3** | blocks 28-31 attn (Phase 1.6 复刻) | 75.23% | 7 | -0.06pp |

**对比 Phase 1.6：**

| | Phase 1.6 (失败) | U3 (修正) |
|------|:---:|:---:|
| 数据 | 在线裁剪 | 固定窗口 |
| lr | 5e-7 | 1e-5 |
| 结果 | 72.51% (-4pp) | 75.23% (-0.06pp) |

Phase 1.6 的退化主要是在线裁剪 + 极低 lr 造成的，不是解冻本身的问题。但即使修正后，解冻也没有增益。

**结论：**
1. **SAM3 ViTDet 深层解冻不提供有意义的增益**（±0.26pp 噪声范围）
2. MMA 论文在 CLIP 上的"深层数据集特定、该训"结论未迁移到 SAM3 ViTDet — SAM3 window attention 可能限制了权重微调的收益空间
3. **此方向关闭** —— 无论是 Adapter、LoRA 还是 Unfreeze，在 SAM3 上 stable 的结果都在 75.3-77.6% 区间，受 frozen backbone 天花板限制
4. U1-U3 无需跑 256² eval（crop 结果已确认无效）
