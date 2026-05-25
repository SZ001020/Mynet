# Plan7: 多模态结构先验注入

> 日期：2026-05-13
> 状态：A/B/C3/D1 已完成；D1 多尺度未超过单尺度软logit基线（77.33 vs 77.55, -0.22pp）
> 当前整图最佳：Plan7-A，软logit eval `OA=88.20`, `mIoU=77.55`（旧per-patch argmax: 77.14）
> 下一步：D2 Learnable Spatial Prompts（可选）或 E 联合训练
> 
> **阶段来源标注**：A/B — 原创设计 | C — 受 SAM-HQ 论文启发 | D — 受 TASAM 论文启发

---

## 1. 核心判断

Plan6 的 Phase 1/1.5/1.6 已经说明：单纯增加 adapter、LoRA、训练轮数或在线裁剪，严格全局结果最高到 `76.57`。这已经超过 MFNet Frozen SAM1 的 `75.11`，但明显低于 MFNet 标准 adapter，说明当前 frozen SAM3 路线仍有平台期。

Plan7 不继续盲目堆 adapter，而是补充 SAM3 遥感分割缺少的结构信息：

- RGB 负责纹理和语义上下文。
- DSM 负责高度和地物几何。
- 结构 prompt 负责边界、坡度、高频位置先验（A/B 阶段）。
- decoder 残差修正负责精细边界修复（C 阶段，受 SAM-HQ 启发）。
- 多尺度测试增强负责消除 ViT patch 对齐偏差（D 阶段，受 TASAM 启发）。

第一阶段只验证最可能有效且噪声最低的先验：**DSM edge / slope**。RGB FFT 高频暂不作为 A 阶段主线，因为它容易增强树冠、阴影、屋顶纹理等噪声。

**C 阶段的灵感来源（SAM-HQ）**：SAM-HQ 证明，在 frozen SAM decoder 端加 HQ-Output Token + Global-local Feature Fusion（<0.5% 参数增量），可以在保留 zero-shot 能力的同时大幅提升边界质量（DIS mBIoU +17.6）。其核心范式——不替换 SAM 输出，训练轻量模块预测"SAM 漏掉了什么"后在 logit 层面做残差加法——是通用的。C 阶段将此范式迁移到 SAM3 + MFNetDecoder pipeline。

**D 阶段的灵感来源（TASAM）**：TASAM 的 MS-SAM 模块通过对输入图像做 0.5×/1.0×/2.0× 多尺度缩放、分别过 frozen SAM encoder 后做 cross-attention 融合，在 LoveDA 上独立贡献 +2.1pp mIoU（Table III 消融）。其底层原理是改变 ViT patch grid 和地物的对齐关系——消除固定分辨率下 patch 边界对地物特征表示的割裂。D 阶段将此思路简化为纯推理时的多尺度测试增强（不训练），并加入 TASAM 的 learnable spatial prompt 注入 decoder 作为可选扩展。

---

## 2. 整体路线

### 2.1 权重继承记录规则

所有阶段如果继承历史最优 checkpoint，必须同时写入本计划和 `model_registry.py`：

- `init_from`：实际初始化 checkpoint。
- `lineage_type`：`formal_ablation` 表示公平消融，`continuation` 表示继承最优继续冲分。
- `comparison_role`：说明该结果能和哪些阶段直接比较。

正式消融必须从同一个父 checkpoint 出发。若为了节省时间从某阶段 best 继续训练，结果只能作为 continuation，不作为 prompt 形态公平对比。

| 阶段 | 目标 | 来源 | 状态 |
|------|------|------|------|
| A | Phase1 full + DSM edge/slope prompt，验证结构先验有效性 | 原创 | **已完成** |
| B | DSM prompt 形态消融：slope/edge/curvature/roughness | 原创 + TASAM 验证 | **已完成** |
| D1 | TASAM 多尺度测试增强，Plan7-A best 上做零训练评估 | **TASAM** | **已完成（多尺度无增益）** |
| C3 | SAM-HQ 残差边界修正：Global-Local Fusion + 残差头 | **SAM-HQ** | **已完成（持平，无增益）** |
| C4 | ~~C3 + prompt dropout~~ | — | C3 未超过基线，跳过 |
| D2 | Learnable Spatial Prompts 注入 decoder | **TASAM** | 可选，C3/C4 有效后再考虑 |
| E | 使用 A/B/C/D 最优配置做 Vaihingen + Potsdam 联合训练 | 原创 | 暂缓 |
| F | 最后 4 层 attention 极小 LR 解冻（最后手段） | 原创 | 可选 |

与 TASAM 的对应关系（验证现有方向 + 区分新方向）：

| TASAM 模块 | Plan7 已有对应 | 关系 |
|-----------|--------------|------|
| TA-Adapter（门控 DSM 融合） | A 阶段 In-ViT gated fusion | ✅ 已有，in-ViT > TASAM 的 late fusion |
| MS-SAM（输入多尺度） | D1: 多尺度测试增强 | 🆕 新增，纯推理时，零训练 |
| TP-Prompt（时序 prompt） | 无（无时序数据） | ✗ 不适配 |
| Learnable spatial prompts | D2: 注入 decoder 的空间 prompt | 🆕 新增，可选扩展 |
| DEM 衍生特征（roughness/aspect） | B3 已含 curvature | ✅ 已有，TASAM 提供外部验证 |

当前推进规则：

- A/B 已完成，`slope+edge` 是当前最优 prompt；不再扩大 prompt 形态搜索。
- 下一步先做 D1 多尺度评估，因为它零训练、能快速判断 patch-grid 对齐是否仍是瓶颈。
- D1 后做 C3 完整残差头；不先跑 C1/C2 矩阵，避免消融过多消耗训练时间。
- 只有 C3 超过 Plan7-A 或明显提升 car/building 边界时，才启动 C4 prompt dropout。
- D2 spatial prompt 与 C 阶段同属 decoder 侧增强，只有 C3/C4 证明 decoder 侧仍有收益时才实现。

---

## 3. A 阶段：DSM Edge/Slope Prompt

### 3.1 设计目标

A 阶段只做一个最小闭环：

```text
RGB + DSM
  → Phase1 full MMAdapter
  + DSM edge/slope prompt
  → MFNetDecoder
```

不做 RGB FFT，不做 Potsdam 联合训练，不解冻 attention。这样可以明确判断：**DSM 结构先验本身是否有效**。

### 3.2 Prompt 构造

从 DSM 生成两个结构通道：

```text
slope = sqrt(dx^2 + dy^2)
edge  = abs(laplacian(dsm)) 或 Sobel magnitude
```

输入维度：

```text
dsm_prompt: B x 2 x H x W
```

归一化策略：

- DSM 仍按 tile min-max 到 `[0,1]`。
- `slope` 和 `edge` 在每个 crop 内做 robust normalize：除以 `p95 + eps`，再 clamp 到 `[0,1]`。
- 不使用 hard threshold，避免丢掉弱边界。

### 3.3 注入方式

新增 `PromptEncoder`，结构与 DSM encoder 保持轻量：

```text
dsm_prompt (2ch)
  → Conv 3x3, stride=2, 32
  → Conv 3x3, stride=2, 64
  → Conv 3x3, stride=2, 128
  → Conv 3x3, stride=2, 128
  → Conv 1x1, 1024
  → BHWC tokens
```

在每个 MMAdapter block 中增加 prompt adapter：

```python
rgb_ada = rgb_mlp_adapter(xn)
dsm_ada = dsm_mlp_adapter(yn)
prompt_ada = prompt_mlp_adapter(pn)

x = x + mlp_x + w_rgb * rgb_ada + w_dsm * dsm_ada + w_prompt * prompt_ada
```

权重使用 softmax gate，而不是多个 sigmoid：

```python
w_rgb, w_dsm, w_prompt = softmax(gate_logits)
```

这样避免三个分支权重相互叠加导致 residual 过大。

### 3.4 训练配置

基线 checkpoint：

```text
/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

继承关系：

```text
init_from = Plan6 Phase1 full DSM self-attention best
lineage_type = formal_ablation
comparison_role = 与 Plan6 Phase1 full 对比，验证 edge+slope prompt 是否有效
```

建议先跑短训：

| 项目 | 配置 |
|------|------|
| 数据 | Vaihingen only |
| 数据加载 | 在线随机裁剪 |
| batch | 2 |
| epoch_steps | 1000 |
| epochs | 8 |
| init | Phase1 full best |
| backbone | frozen |
| DSM attention | full + checkpoint |
| adapter lr | `5e-5` |
| dsm/prompt lr | `2.5e-5` |
| decoder lr | `5e-5` |
| loss | structure_loss |

为什么不用 20-30 epoch：

- 之前所有路线第 1-5 epoch 已经暴露趋势。
- 如果 prompt 有效，短训应该能超过 Phase1 crop best `76.86` 或接近。
- 长训容易把 prompt 当噪声记忆，不能先验判断。

### 3.5 验收

A 阶段先看 crop validation，但最终只认 256² 整图。

短训后判断：

| 结果 | 动作 |
|------|------|
| crop best `<76.8` | 不做整图评估，A 阶段失败 |
| crop best `76.8-77.2` | 做一次整图评估 |
| crop best `>77.2` | 做整图评估，并优先延长训练到 15 epoch |

历史验收标准（A 已完成，实际结果 `mIoU=77.14`）：

| 结果 | 动作 |
|------|------|
| `mIoU <=76.57` | 未超过 Plan6，prompt 无效 |
| `76.57-77.0` | 弱有效，只做最小 B 消融 |
| `77.0-77.5` | 有效，进入 D1/C3 |
| `>=77.5` | Plan7 主线强成立，优先做 D1/C3 联合验证 |

---

## 4. B 阶段：DSM Prompt 形态消融

B 阶段只回答一个问题：A 阶段的提升到底来自哪种 DSM 结构 prompt。当前不引入 RGB high-pass，不做 text prompt，不做 Potsdam 联合训练，不解冻 backbone。

### 4.1 对照关系

```text
B0: Plan7-A edge+slope，已有结果，不重跑
B1: slope_only
B3: slope+edge+curvature
```

严格全局评估对照：

| 模型 | 256² OA | 256² mIoU | 说明 |
|------|---------|-----------|------|
| Plan6 Phase1 full | `87.27` | `76.57` | DSM full attention 基线，严格全局口径 |
| Plan7-A edge+slope | `87.75` | `77.14` | 当前 Plan7 最好 |
| Plan7-B1 slope_only | `87.26` | `76.49` | 不如 A，说明 edge 有效 |
| Plan7-B3 slope+edge+curvature | `87.33` | `76.49` | 不如 A，curvature 无增益 |

### 4.2 B1：Slope Only

只保留 DSM Sobel slope：

```text
dsm_prompt: B x 1 x H x W
slope = sqrt(dx^2 + dy^2)
```

目的：

- 判断 A 阶段的收益是否主要来自地形梯度。
- 去掉 Laplacian edge 噪声，观察 grass/tree/road 是否更稳。

代码目录：

```text
RS-SAM3-p7/phase_b_prompt_ablation/b1_slope_only/
```

### 4.3 B3：Slope + Edge + Curvature

在 A 的 slope+edge 上加入局部 curvature/roughness：

```text
dsm_prompt: B x 3 x H x W
slope = Sobel magnitude
edge = abs(Laplacian DSM)
curvature = avg_pool(edge, kernel=5)
```

目的：

- 检查更平滑的地形结构先验能否改善 building/car 边界。
- 避免直接加入 absolute height 带来的跨 tile 分布不稳定。

代码目录：

```text
RS-SAM3-p7/phase_b_prompt_ablation/b3_slope_edge_curvature/
```

### 4.4 训练配置

B 阶段必须固定除 prompt 通道外的所有变量：

```bash
--dataset vaihingen
--epochs 8
--batch 2
--epoch-steps 1000
--resolution 1008
--lr 5e-5
--dsm-lr 2.5e-5
--prompt-lr 2.5e-5
--dsm-attn-mode full
--checkpoint-attn
--init-from /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
```

继承关系：

```text
init_from = Plan6 Phase1 full DSM self-attention best
lineage_type = formal_ablation
comparison_role = 与 Plan7-A edge+slope 做 prompt 形态公平对比
```

如果从 Plan7-A best 继续训练，必须标记为：

```text
init_from = Plan7-A best
lineage_type = continuation
comparison_role = 只能判断是否能继续冲分，不能作为 B 阶段公平消融
```

### 4.5 验收标准

只认 256² 滑窗评估：

| 结果 | 动作 |
|------|------|
| B1/B3 都 `<77.14` | A 的 edge+slope 已是当前最优，停止 B 阶段 |
| 任一结果 `77.14-77.5` | 弱提升，保留最佳 prompt，但不扩大 B |
| 任一结果 `>=77.5` | prompt 形态有效，先做 D1，再做 C3 |
| 任一结果 `>=78.0` | Plan7 prompt 路线强成立，优先验证 D1+C3 组合 |

不做 B2 absolute height，除非 B1/B3 均失败且错误主要集中在 building/car。

### 4.6 B 阶段结论

B 阶段正式消融已完成，全部从同一个父 checkpoint 初始化：

```text
init_from = /root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt
lineage_type = formal_ablation
```

最终结果：

| 阶段 | Prompt | crop best | best epoch | 全局 OA | 全局 mIoU | mRecall |
|------|--------|-----------|------------|---------|-----------|---------|
| A | slope + edge | `76.87` | 3 | `87.75` | `77.14` | `87.78` |
| B1 | slope only | `76.77` | 6 | `87.26` | `76.49` | `86.25` |
| B3 | slope + edge + curvature | `76.62` | 1 | `87.33` | `76.49` | `87.31` |

判断：

- `slope only` 低于 A，说明 DSM edge 对整体 mIoU 有实际贡献。
- 加入 `curvature/roughness` 没有提升，反而压低 tree/car 和整体 mIoU。
- 当前最佳 DSM prompt 保持 A 阶段的 `slope + edge`。
- B 阶段不再继续扩大 slope/edge/curvature 组合消融。

---

## 5. C 阶段：SAM-HQ 残差边界修正 [来源：SAM-HQ 论文]

### 5.1 设计动机

SAM-HQ 的核心范式——在 frozen 模型 decoder 端用极轻量模块做 logit-level 残差修正——经实验验证在保留泛化能力的同时显著提升边界质量。C 阶段将此范式迁移到 SAM3 + MFNetDecoder 场景。

关键设计原则（来自 SAM-HQ 消融实验）：

| SAM-HQ 发现 | C 阶段对应设计 |
|-------------|---------------|
| 不动 encoder，动 decoder | SAM3 ViTDet 全部 frozen |
| HQ-Token + MLP 做残差 | 轻量残差头预测修正 logits |
| Early+Late ViT + Decoder 特征融合 | Block 6 + Block 32 + MFNetDec b3 中间层 |
| 逐元素相加融合 > FPN | 直接 sum，不做额外 pyramid |
| 44K 小数据集 4h 训练 | Vaihingen 12 张图 + 增强 |
| 混合 prompt 训练防过拟合 | 随机 dropout DSM prompt 30% |

### 5.2 架构修改

在现有 Plan7-A 模型输出端增加两个轻量组件：

```
当前 Plan7-A:
  SAM3 ViTDet (frozen, 32 blocks)
    ├→ feat@block 8  ──┐
    ├→ feat@block 16 ──┤
    ├→ feat@block 24 ──┼→ Pyramid4Scale → MFNetDecoder → logits → Ŷ
    ├→ feat@block 32 ──┘

C 阶段增加:
  SAM3 ViTDet (frozen, 32 blocks)
    ├→ feat@block 6  ─────────────────┐  ← 新增：第一个 global attn 之后的早期特征
    ├→ feat@block 8  ──┐               │
    ├→ feat@block 16 ──┤               │
    ├→ feat@block 24 ──┼→ Pyramid4Scale│→ MFNetDecoder → logits ─┬→ logits_final → Ŷ
    ├→ feat@block 32 ──┘               │                          │
    │                      b3_feat ────┤  ← Decoder 1/16 中间层   │
    │           (block 32 feat) ───────┤  ← 复用，不需额外抽取     │
    └──────────────────────────────────┘                          │
         Global-Local Feature Fusion                              │
         └→ [ConvTr + Conv 融合] → HQ-Features → ResidualHead ────┘
                                                      ↑
                                              残差 logit 加法
```

#### 组件 1：Global-Local Feature Fusion

```python
# 取三路特征
early_feat  = vit_encoder.block6_output   # B×1024×64×64, 第一个 global attn 后
late_feat   = backbone_fpn[-1]            # B×256×64×64, 复用已有输出
decoder_mid = mfnet_decoder.b3_output     # B×64×H/16×W/16, decoder 中间层

# 对齐到统一分辨率 (decoder 输出分辨率的 1/4)
compress_early = nn.Sequential(           # SAM-HQ: compress_vit_feat
    ConvTranspose2d(1024, 256, 2, 2),    # 64→128
    LayerNorm2d(256), GELU(),
    ConvTranspose2d(256, 64, 2, 2))      # 128→256

compress_late = nn.Sequential(            # SAM-HQ: embedding_encoder
    ConvTranspose2d(256, 128, 2, 2),     # 64→128
    LayerNorm2d(128), GELU(),
    ConvTranspose2d(128, 64, 2, 2))      # 128→256

compress_decoder = nn.Sequential(          # SAM-HQ: embedding_maskfeature
    Conv2d(64, 128, 3, 1, 1),
    LayerNorm2d(128), GELU(),
    Conv2d(128, 64, 3, 1, 1))

# HQ-Features: 简易逐元素相加（SAM-HQ Table 3 验证 sum > FPN）
hq_features = compress_early(early_feat) + compress_late(late_feat) + compress_decoder(decoder_mid)
```

**为什么 block 6 而不是更早的层**：SAM3 ViTDet 有 32 个 block，global attention 在 blocks [7, 15, 23, 31]。Block 6 是最后一个纯 window attention block——它保留了最多的局部边界信息，尚未被 global attention 的远距离交互稀释。这对应 SAM-HQ 在 ViT-L 24-block 中取 block 6（第一个 global attn 前）的做法。

#### 组件 2：残差修正头

```python
# 轻量残差头：HQ-Features → residual logits
self.residual_head = nn.Sequential(
    ConvBNReLU(64, 64, kernel_size=3),     # 保持轻量
    nn.Dropout2d(0.1, inplace=False),
    Conv2d(64, num_classes, kernel_size=1) # 输出残差 logits
)

# 推理：与 SAM-HQ 完全一致的 logit 加法
logits_main = decoder(original_feats)           # [B, 5, H/4, W/4]
logits_residual = residual_head(hq_features)    # [B, 5, H/4, W/4]
logits_final = logits_main + logits_residual
```

残差头参数：64×64×3×3 + 64×5×1×1 ≈ **37K**，几乎无增量。

#### 组件 3：训练时随机 DSM Prompt Dropout

```python
# 受 SAM-HQ "混合 prompt 训练" 启发
# 训练时 30% 概率将 DSM prompt 全部置零
if training and random.random() < 0.3:
    mm_state["prompt_tokens"] = torch.zeros_like(mm_state["prompt_tokens"])
    # decoder 仍正常工作，残差头学习从 RGB + 原始 DSM 特征中提取边界
```

目的：让残差头不过度依赖 DSM prompt，在 DSM 不可用时仍能依赖 RGB 特征修正边界。

### 5.3 执行策略

C 阶段不再先铺开 C1/C2/C3/C4 全矩阵。B 阶段已经证明 prompt 形态继续搜索收益低，当前更需要验证 SAM-HQ 范式本身是否有效。

| 优先级 | 变体 | 说明 | 动作 |
|--------|------|------|------|
| 1 | **C3** | 完整残差头：block 6 + block 32 + decoder mid | 先实现、先训练 |
| 2 | **C4** | C3 + 随机 DSM prompt dropout | 仅 C3 有收益后启动 |
| 3 | C1/C2 | decoder-only / decoder+late 局部消融 | 仅需要写论文归因时补跑 |

C3/C4 必须从 Plan7-A best 出发，标记为 `formal_ablation`。如果后续从 C3 best 继续训练，只能标记为 `continuation`。

### 5.4 训练配置

| 项目 | 配置 |
|------|------|
| 数据 | Vaihingen only，在线随机裁剪 |
| batch | 2 |
| epoch_steps | 1000 |
| epochs | 8（C3 短训）/ 15（C3 有效后完整验证） |
| init | Plan7-A best |
| backbone | **全部 frozen**（包括 SAM3 ViTDet） |
| dsm_attn_mode | full + checkpoint |
| decoder lr | `5e-5` |
| residual head lr | `1e-4`（新模块可稍高） |
| dsm/prompt lr | `2.5e-5` |
| loss | structure_loss |

继承关系：

```text
init_from = Plan7-A best
lineage_type = formal_ablation（C3/C4 从同一父节点出发）
comparison_role = 验证 SAM-HQ 残差范式在 SAM3+MFNetDecoder 上的迁移有效性
```

### 5.5 验收标准

C3 短训（8 epoch）已于 2026-05-13 完成：

```text
Run: plan7_c3_residual_vaihingen_20260513_004439
Checkpoint: best_model.pt (epoch 2, crop val mIoU=77.15%)
256² 整图: OA=87.67%, mIoU=77.05%

Delta vs Plan7-A (77.14): -0.09pp (噪声级)
  road:  -0.42pp  bldg: -0.54pp  grass: +0.83pp  tree: +0.32pp  car: +0.34pp
```

| 结果 | 动作 |
|------|------|
| C3 < C0（77.14）| **确认：残差修正范式在 SAM3 上无增益** |
| ~~C3 >= 77.5~~ | 未达到 |

结论：SAM-HQ 的残差修正范式在 SAM3 + MFNetDecoder 上未能迁移成功。可能原因：
1. MFNetDecoder 的 3 个 GLA block + FeatureRefinementHead 已经做了足够强的特征精炼，残差头成为冗余
2. SAM-HQ 的 HQ-Token 机制在 SAM1 的 2 层 Transformer decoder 中有效，但 SAM3 用的是不同架构
3. 37K 参数的残差头太小，不足以学习有意义的修正

C4（prompt dropout）跳过——C3 未超过基线，无需测试 dropout。

---

## 6. D 阶段：TASAM 多尺度测试增强 + Learnable Spatial Prompts [来源：TASAM 论文]

### 6.1 设计动机

TASAM 的 MS-SAM 对输入图像做 0.5×/1.0×/2.0× 缩放后分别过 SAM encoder，通过 cross-attention 融合多尺度特征（Table III 消融中独立贡献 +2.1pp mIoU）。其底层原理是改变 ViT 16×16 patch grid 和地物的物理对齐关系——

SAM3 ViTDet 在 1008² 分辨率下，每个 patch 覆盖 1.44m×1.44m 物理地面（9cm GSD）。Vaihingen 汽车（2-5m）只占 1.5-3.5 个 patch。当一辆车跨越 patch 边界时，patch embedding 将车辆语义割裂到两个 token 中，window attention 虽能交互但 patch 边界上的语义已被线性投影混合。改变输入尺度 → 改变 patch grid 的物理对齐 → 降低单尺度 patch 边界偏差。

TASAM 的 learnable spatial prompts（从 DEM 特征通过 MLP 动态生成 k 个 prompt tokens 送入 decoder）提供了另一种空间信息注入路径，和 Plan7-A 的 in-ViT prompt 互补：A 阶段在 ViT 内注入，D 阶段在 decoder 端注入。

### 6.2 子任务

| 子任务 | 内容 | 训练成本 | 推理成本 |
|--------|------|:---:|:---:|
| **D1** | 推理时 2 尺度测试增强（0.75× + 1.0×） | **零**（不改训练） | <2× 推理时间 |
| D2 | Learnable spatial prompts 注入 MFNetDecoder | ~0.1M 参数 | 几乎无 |
| D3 | B3 已有 curvature，加 roughness/aspect 外部验证 | 仅改 dsm_prompt.py | 无 |

D1 是当前下一步主项（零训练、低风险，先在 Plan7-A best 上验证）。D2 是可选扩展，只有 C3/C4 证明 decoder 侧增强有效后才实现。D3 已由 B3 的 curvature 结果覆盖，不再单独启动。

### 6.3 D1：推理时多尺度测试增强

#### 原理

```
单尺度（当前）:
  image @ 1008² → patch grid 63×63 → ViT → logits → Ŷ
  问题：patch 边界对齐偏差——同一辆车因与 grid 相对位置不同获得不同特征

双尺度（D1）:
  image @ 1008² (1.00×) → patch grid 63×63 → logits_1.0
  image @  756² (0.75×) → pad to 1008² → 有效 patch grid ~47×47 → logits_0.75
  logits_final = (upsample(logits_0.75) + logits_1.0) / 2.0
```

0.75× 下每个 patch 覆盖 1.92m（比 1.0× 的 1.44m 多 33% 物理范围），同一辆车的 patch 覆盖模式不同。两个预测取均值 = 两种 patch 对齐方式的 ensemble，消除单尺度的对齐偏差。

**不加 1.25× 的原因**：
- 放大后分辨率 > 1008² 需要额外 crop，工程复杂度增加
- 推理更慢（patch 数量翻倍）
- TASAM 的 0.5× 替换为 0.75× 更适合 SAM3：700² → pad to 1008² 损失小于 500² → pad to 1008²

#### 实现

在 eval.py 中增加 `--ms-scales` 参数：

```python
def predict_multiscale(model, image, dsm, scales=[0.75, 1.0]):
    """
    多尺度推理：对每个 scale 缩放到 1008² 后过模型，logits 取均值。
    所有尺度都利用已有 256² 滑窗逻辑——多尺度是滑窗的外层循环。
    """
    logits_list = []
    for scale in scales:
        if scale == 1.0:
            logits = model(image, dsm)  # 已有路径，不额外开销
        else:
            h, w = image.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = F.interpolate(image, (new_h, new_w), mode='bilinear')
            scaled_dsm = F.interpolate(dsm, (new_h, new_w), mode='bilinear')
            # pad to 1008² (SAM3 ViT 固定输入)
            scaled_img = pad_to(scaled_img, 1008)
            scaled_dsm = pad_to(scaled_dsm, 1008)
            logits = model(scaled_img, scaled_dsm)
            # 裁剪回缩放后区域 + 回采样到原始分辨率
            logits = logits[..., :new_h//4, :new_w//4]
            logits = F.interpolate(logits, (h//4, w//4), mode='bilinear')
        logits_list.append(logits)

    return torch.stack(logits_list).mean(dim=0)
```

#### 预期收益与风险

| 场景 | 预期收益 | 说明 |
|------|:---:|------|
| Car IoU | +0.3-0.8pp | 1-4 patch 的物体对对齐最敏感 |
| Building 边界 BIoU | +0.2-0.4pp | 边界区的 patch 割裂改善 |
| Road/Tree/Grass | +0.0-0.1pp | 大区域对 patch 对齐不敏感 |
| 整体 mIoU | +0.1-0.3pp | 小物体权重低，拉低均值 |
| 推理时间 | +80%（≈1.8×） | 0.75× 下 patch 少 44%，补回部分时间 |

**优先验收**：D1 先在 Plan7-A best 上评估。如果 car recall 或 car IoU 没有正向提升，D1 只保留为备选 eval 选项，不进入默认流程。若 D1 有效，再在 C3/C4 最优模型上复测。

### 6.4 D2：Learnable Spatial Prompts 注入 Decoder（可选扩展）

#### 原理

TASAM 用 MLP 从 DEM 特征生成 k 个 learnable spatial prompt tokens 送入 decoder（非 ViT）。当前 Plan7-A 的 prompt_encoder 将 DSM 结构先验注入 **ViT 内部每个 block**，D2 改为注入 **MFNetDecoder 输入端**——两者互补：

```
A 阶段 (in-ViT prompt):
  DSM prompt → ViT internal blocks → 参与 attention 计算
  优势：全局交互，长程依赖

D2 (decoder prompt, TASAM style):
  DSM prompt → MLP → k 个 256-dim spatial tokens → concat to MFNetDecoder b4 input
  优势：轻量（decoder 不用额外 attention），近输出端，效果直接
```

#### 实现

```python
# dsm_prompt.py 中新增
class SpatialPromptDecoder(nn.Module):
    """TASAM-style: DSM → MLP → k learnable spatial tokens for decoder."""
    def __init__(self, dsm_channels=1, num_tokens=4, token_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),        # DSM → 16×16
            nn.Conv2d(dsm_channels, 64, 3, 1, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),          # → 1×1×64
        )
        self.token_gen = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, num_tokens * token_dim)
        )
        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, dsm):
        # dsm: B×1×H×W
        feat = self.encoder(dsm).flatten(1)            # B×64
        tokens = self.token_gen(feat)                   # B×(k*256)
        tokens = tokens.view(-1, self.num_tokens, self.token_dim)  # B×k×256
        return tokens

# model.py 中注入 decoder
spatial_tokens = self.spatial_prompt(dsm)  # B×4×256
# 在 MFNetDecoder.b4 输入端扩展 feature channel
x = self.b4(self.pre_conv(res4) * spatial_modulation(spatial_tokens))
```

#### 预期收益与风险

| 项 | 说明 |
|----|------|
| 参数增量 | ~0.1M |
| 预期收益 | +0.2-0.5pp（如果 decoder 缺乏空间定位信息） |
| 主要风险 | SAM3 内部的 text-conditioned semantic head 可能已经提供了足够的空间信息，spatial tokens 成为冗余 |
| 与 C 阶段的配合 | 残差修正头 + spatial decoder prompt → 双重 decoder 侧增强 |

**验收**：仅当 C3/C4 高于 Plan7-A（`mIoU > 77.14`）或显著提升 car/building 边界时才启动 D2，作为 decoder 侧的第二个增强点。如果 C3/C4 未超过 A，跳过 D2，避免重复实现低收益 decoder 分支。

### 6.5 D1+D2 训练/推理配置

D1 是纯推理时技巧，不训练。如果启动 D2：

| 项目 | D2-only | D1+D2 联合 |
|------|---------|-----------|
| 数据 | Vaihingen only | — |
| batch/epoch_steps/epochs | 同 C（2/1000/8） | — |
| init | C 最优 checkpoint | — |
| backbone | frozen（全部） | — |
| spatial prompt lr | `1e-4` | — |
| D1 启用 | — | eval 时加 `--ms-scales 0.75,1.0` |
| lineage_type | formal_ablation | D1 不需标记（纯推理） |

### 6.6 D1 实验结果（2026-05-13）

**注意**：D1 切换了 eval 方法——从 per-patch argmax 改为软 logit 积累。单尺度软 logit 基线已优于旧 eval。

```text
旧 eval (per-patch argmax):  Plan7-A mIoU=77.14%  (老基线)
软 logit 单尺度 (1.0×):       mIoU=77.55%  (D1 公平基线)
软 logit 双尺度 (1.0+0.75×):   mIoU=77.33%  (-0.22pp vs 单尺度)

Per-class Δ(双-单): road -0.29, bldg -0.18, grass +0.21, tree -0.03, car -0.81
```

| 结果 | 结论 |
|------|------|
| **D1 多尺度 < 单尺度基线** | **多尺度测试增强对 SAM3+ViTDet 无效** |
| 软 logit 积累 > 旧 per-patch argmax (+0.41pp) | 后续所有 eval 改用软 logit 积累 |

分析：
- car -0.81pp 是最意外的——理论预测小物体最受益于多尺度，实际上 0.75× 下采样让小车更小，边界更模糊
- grass +0.21pp 是唯一正向的类——大区域对尺度不敏感，软 logit 多尺度平均可能减少了噪声
- 整体结论：SAM3 ViTDet 的 patch 对齐偏差在 ISPRS 9cm GSD + 256² 滑动窗口下不构成实际瓶颈。TASAM 的 MS-SAM 在 SAM1+自然图像（COCO/LoveDA）上有效的条件可能与 SAM3+遥感不同

---

## 7. E 阶段暂缓：联合训练

只有 A/B/C/D 证明各阶段有效后，才做 Vaihingen + Potsdam joint training。

联合训练使用 C 阶段最优配置（含残差头 + prompt dropout 如果验证有效）+ D 阶段最优 eval 配置（若 D1 有效则默认启用多尺度评估）。

联合训练必须分别评估：

```text
train V+P -> eval Vaihingen
train V+P -> eval Potsdam
```

如果只提升 crop validation，不提升两个数据集的 256² 整图结果，则不算成功。

---

## 8. F 阶段暂缓：深层 Attention 解冻

F 阶段不是必选项。只有 A-E 仍卡在 `77` 左右时，才尝试最后 4 层 attention 解冻。

默认策略：

```text
unfreeze blocks 28-31 attn.qkv / attn.proj
attn_lr = 5e-7
MLP 和浅层 backbone 继续 frozen
```

如果 A-E 已经能到 `78+`，F 阶段可以跳过。

---

## 9. 文件结构

```text
RS-SAM3-p7/
├── phase_a_dsm_prompt/                     # A 阶段 [原创]
│   ├── dsm_prompt.py
│   ├── mm_adapter_vit.py
│   ├── model.py
│   ├── train_a.py
│   └── eval.py
├── phase_b_prompt_ablation/                # B 阶段 [原创 + TASAM 验证]
│   ├── b1_slope_only/
│   └── b3_slope_edge_curvature/
├── phase_c_residual_boundary/              # C 阶段 [来源：SAM-HQ]
│   ├── c3_full/
│   └── c4_dropout/                         # 仅 C3 有效后建立
├── phase_d_multiscale_spatial/             # D 阶段 [来源：TASAM]
│   ├── d1_multiscale_eval/
│   │   └── eval_ms.py                      # 多尺度推理 eval 脚本（独立，不改模型）
│   └── d2_spatial_prompt/                  # 可选扩展
│       ├── spatial_prompt_decoder.py        # TASAM-style learnable spatial tokens
│       ├── model.py
│       └── train_d2.py
└── run_plan7_b_all_and_shutdown.sh
```

C 阶段先只实现 `c3_full/`，共享文件为 `hq_fusion.py`（`GlobalLocalFusion`）+ `residual_head.py`（`ResidualCorrectionHead`）。C1/C2 只作为论文归因备选，不进入当前执行队列。

D1 是纯 eval 脚本，只创建 `d1_multiscale_eval/`，不创建新的训练模型分支。D2 需要独立子目录（含 `spatial_prompt_decoder.py`）。

---

## 10. 当前结论

```text
Plan6 Phase1 full:      76.57  (严格全局基线)
Plan7-A edge/slope:     77.14  (+0.57, 当前 Plan7 最优)
Plan7-B slope only:     76.49  (未超过 A)
Plan7-B +curvature:     76.49  (未超过 A)
Plan7-D1 (TASAM):       下一步 (多尺度测试, 0 训练参数)
Plan7-C3 (SAM-HQ):      D1 后启动 (完整残差修正头, <40K 参数)
Plan7-C4 dropout:       条件启动 (仅 C3 有效后)
Plan7-D2 spatial:       条件启动 (仅 C3/C4 有效后)
```

A/B 已确认 DSM `slope + edge` 是当前最优结构 prompt；`slope only` 和 `curvature` 都没有超过 A。后续执行顺序调整为：

1. **D1 先做**：零训练成本，先判断 patch-grid 对齐是否还能贡献收益。
2. **C3 再做**：完整 SAM-HQ 残差头，直接验证 decoder 残差范式，不铺开 C1/C2。
3. **C4/D2 条件启动**：只有 C3 或 D1 确认有效后，再做 prompt dropout 或 decoder spatial prompt。

两条路线互补：C 改模型（训练时），D 不改模型（推理时）。两条路线都保持 frozen backbone 约束。

如果 C+D 联合能将 mIoU 推至 77.5-78.0，说明在 frozen 范式下 decoder 侧增强 + 多尺度测试是两个有效的低成本突破方向。此时再进入 E 阶段（联合训练）扩大数据规模。如果 C+D 仍卡在 77.14 附近，frozen SAM3 ViTDet 的瓶颈被充分确认，后续实验需要重新审视 Plan6 的结论——是否需要在更大的数据集（LoveDA 等）上用相同架构训练。
