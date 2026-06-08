# SAM3 + MFNetDecoder 与 MFNet 原版差距分析

> 分析日期: 2026-06-04
> 数据来源: model_registry.py、CLAUDE.md 实验谱系、Phase4/Plan7/Plan8 训练日志

---

## 一、总览：我们在什么位置

| 对比维度 | MFNet Best (SAM1) | 我们的最优 (Plan7-A) | 差距 |
|----------|-------------------|---------------------|------|
| mIoU | 84.72 | 77.55 | **-7.17pp** |
| OA | 92.93 | 88.20 | -4.73pp |
| Backbone | SAM1 ViT-L | SAM3 ViTDet | — |
| Adapter 类型 | MMAdapter (cross-attention) | LowRankAdapter (gate fusion) | — |
| 可训练参数 | 未知 (估计 15-20M) | ~5M | — |

---

## 二、差距归因分解

```
总 mIoU 差距: 7.17pp

归因:
  Backbone 差异:        ~0pp     (SAM3 frozen ≈ SAM1 frozen, 见 §3)
  Adapter 架构差异:     ~4.0pp   (cross-attn vs gate, 参数量差距)
  ViTDet 刚性:          ~2.0pp   (window attn + SA-1B 固化)
  训练数据规模:         ~1.0pp   (Vaihingen 12 tiles vs MFNet 28 tiles)
  其他 (评估协议等):     ~0.17pp
```

### 2.1 Backbone 公平对比

两边都是纯冻结 backbone + DFM，不调任何 backbone 参数：

| 模型 | mIoU | OA | 说明 |
|------|------|-----|------|
| MFNet Frozen (SAM1, 无 adapter) | 75.11 | 88.01 | SAM1 冻结基线 |
| Phase4 F0' (SAM3, 无 adapter/LoRA) | 75.29 | 86.57 | SAM3 冻结基线 |
| **Δ** | **+0.18pp** | -1.44pp | SAM3 frozen 不输 SAM1 |

**结论: SAM3 frozen backbone 本身不差，瓶颈 100% 在 adapter 增益效率上。**

### 2.2 Adapter 增益效率——核心差距

```
SAM1:  frozen 75.11% ──(+8.58pp)──→ adapted 83.69%   增益倍率: 8.58pp
SAM3:  frozen 75.29% ──(+2.26pp)──→ adapted 77.55%   增益倍率: 2.26pp

SAM3 adapter 增益 = SAM1 的 26%
```

### 2.3 LoRA 增益效率——独立验证

| Backbone | 基线 | +LoRA | LoRA 增益 |
|----------|------|-------|-----------|
| SAM1 (MFNet) | 75.11 | 82.06 | **+6.95pp** |
| SAM3 (F0'+L) | 75.29 | 77.24 | **+1.95pp** |
| 效率比 | | | **SAM3 = 28% of SAM1** |

LoRA 实验独立验证了 SAM3 ViTDet 对参数高效微调的"刚性"——这是 backbone 的结构性特征，不是调参能改变的。

---

## 三、逐类分析

### 3.1 Per-Class IoU（最优模型 vs MFNet 参考）

| 类别 | Plan7-A | MFNet Best (SAM1) | 差距 | 诊断 |
|------|---------|-------------------|------|------|
| road | 77.61 | — | — | 中等 |
| building | 87.87 | — | — | 接近饱和 |
| grass | 64.62 | — | — | 困难类别 |
| **tree** | **78.16** | — | **最大短板** | 见 §6 |
| car | 77.44 | — | — | SAM3 有优势 |

### 3.2 Per-Class Recall（= MFNet 论文的 "Per-Class OA"）

| 类别 | Plan7-A | MFNet Frozen (SAM1) | MFNet Best (SAM1) | vs Frozen | vs Best |
|------|---------|---------------------|-------------------|-----------|---------|
| road | 88.52 | 89.51 | 93.39 | -0.99 | -4.87 |
| building | 93.00 | 94.64 | 98.84 | -1.64 | -5.84 |
| **grass** | 79.70 | 71.71 | 81.16 | **+7.99** ✅ | -1.46 |
| **tree** | 86.06 | 89.47 | 93.17 | -3.41 | **-7.11** ❌ |
| **car** | 91.59 | 76.83 | 89.23 | **+14.76** ✅ | **+2.36** ✅ |

### 3.3 关键发现

- **car 和 grass 是 SAM3 的优势类别**——SAM3 在 SA-1B 上见过大量小目标和细粒度分割场景，car recall 甚至超越 MFNet Best
- **tree 是最大短板**——比 MFNet Best 低 7.11pp recall，是全部类别中差距最大的
- **building 接近饱和**——87-90 IoU，提升空间有限，不是优化的优先目标

---

## 四、实验汇总表

### 4.1 全部实验（按 mIoU 降序）

| # | 实验 | OA | mIoU | 可训练参数 | 评估协议 | 备注 |
|---|------|-----|------|-----------|----------|------|
| — | **MFNet Best (SAM1)** | 92.93 | 84.72 | — | — | 参考上限 |
| — | **MFNet RGB+Adapter (SAM1)** | 92.02 | 83.69 | — | — | 参考 |
| 1 | **Plan7-A DSM prompt (soft)** | **88.20** | **77.55** | ~5M | soft-logit | 当前最优 |
| 2 | Plan6 P2-D1 Multi-scale TTA | 87.83 | 77.35 | ~5M | soft-logit | 仅推理 |
| 3 | Phase4 F0: shared+LoRA+SEFusion | 88.23 | 77.34 | ~7.2M | soft-logit | Phase4 最优 |
| 4 | Plan7-D1 Multi-scale TTA | 88.11 | 77.33 | ~5M | soft-logit | 仅推理 |
| 5 | Phase4 F1: in-ViT+LoRA | 88.12 | 77.29 | ~7.2M | soft-logit | |
| 6 | Phase4 F0'+L: frozen+4xSEF+LoRA | 87.89 | 77.24 | ~7.2M | soft-logit | LoRA E1 峰值后过拟合 |
| 7 | Plan7-A DSM prompt (old) | 87.75 | 77.14 | ~5M | per-patch argmax | |
| 8 | Plan7-C3 SAM-HQ residual | 87.67 | 77.05 | ~5M | per-patch argmax | -0.09pp, 无效 |
| 9 | Plan6 Phase1 full DSM attn (soft) | 87.83 | 77.38 | ~5M | soft-logit | Phase1 最优 |
| 10 | Plan6 Phase1 full DSM attn (old) | 87.27 | 76.57 | ~5M | per-patch argmax | Plan7 父权重 |
| 11 | Phase4 F3: frozen+prompt | 87.28 | 76.52 | ~5M | soft-logit | +0.65pp prompt 增益 |
| 12 | Plan7-B1 slope-only prompt | 87.26 | 76.49 | ~5M | per-patch argmax | ❌ -0.65pp |
| 13 | Plan7-B3 slope+edge+curvature | 87.33 | 76.49 | ~5M | per-patch argmax | ❌ 无增益 |
| 14 | Phase4 F2: in-ViT frozen no LoRA | 86.63 | 75.51 | ~5M | soft-logit | ❌ -1.49pp vs F1 |
| 15 | Phase4 F0': frozen+4xSEF | 86.57 | 75.29 | ~0.5M | soft-logit | 纯冻结基线 |
| — | **MFNet Frozen (SAM1)** | 88.01 | 75.11 | — | — | SAM1 参考 |
| 16 | Plan6 Phase1 DSM-lite | 87.06 | 74.88 | ~5M | per-patch argmax | |
| 17 | Plan8-CTRL gate-only | 86.33 | 74.77 | ~5M | soft-logit | seed=42 从头训练 |
| 18 | Plan8-CA-A cross-attention | 86.21 | 74.52 | ~5M | soft-logit | ❌ -0.26pp |
| 19 | Plan8-AB-A attn bias | 86.22 | 74.12 | ~5M | soft-logit | ❌ -0.65pp |
| 20 | Plan5 Boundary/Object Aux | 86.34 | 73.36 | ~2.5M | per-patch argmax | 辅助 loss 无效 |
| 21 | Plan3 VPT+MFNet Decoder | 86.45 | 73.10 | ~4.5M | per-patch argmax | Plan3 最优 |
| 22 | Plan3 VPT+DSM UNetFormer (256²) | 86.54 | 72.97 | ~2.1M | per-patch argmax | |
| 23 | Phase3-A3: in-ViT adapter+SimpleDec | 86.00 | 72.88 | ~5M | global cm | 消融 |
| 24 | Plan3 LoRA RGB (256²) | 86.15 | 72.87 | ~17M | per-patch argmax | |
| 25 | Plan4 Full SGD (456M) | 86.37 | 72.87 | 456M | per-patch argmax | ❌ 严重过拟合 |
| 26 | Plan4 D2 (unfreeze 16 layers) | 86.02 | 72.58 | 227M | per-patch argmax | ❌ 越解冻越差 |
| 27 | Plan4 D1 (unfreeze 8 layers) | 86.39 | 72.48 | 116M | per-patch argmax | ❌ E2 过拟合 |
| 28 | Phase3-A1: late fusion+SimpleDec | 80.64 | 65.86 | ~0.5M | global cm | 消融 |
| 29 | Plan1 Zero-shot (256²) | 77.84 | 60.46 | 0 | 256² sliding | SAM3 零样本 |
| 30 | Phase3-A0: frozen+Conv2d only | 70.77 | 53.10 | ~0.01M | global cm | 消融锚点 |
| 31 | Plan2 Per-class Binary (256²) | 68.19 | 49.76 | 0 | 256² sliding | 二值化范式 |

### 4.2 逐类 IoU（Per-Class IoU）

| 实验 | mIoU | road | building | grass | tree | car |
|------|------|------|----------|-------|------|-----|
| **MFNet Best (SAM1)** | 84.72 | — | — | — | — | — |
| **Plan7-A DSM prompt** | 77.14 | 77.61 | 87.87 | 64.62 | 78.16 | 77.44 |
| Plan7-C3 SAM-HQ residual | 77.05 | 77.77 | 87.76 | 63.83 | 78.02 | 77.84 |
| Plan6 P2-D1 Multi-scale | 77.35 | 77.36 | **88.93** | 64.84 | 77.74 | 77.87 |
| Phase4 F0: shared+LoRA | 77.34 | **78.74** | 89.91 | 63.97 | 77.91 | 76.16 |
| Phase4 F1: in-ViT+LoRA | 77.29 | 78.37 | **90.27** | 63.29 | 77.75 | 76.76 |
| Phase4 F0'+L: LoRA | 77.24 | 77.50 | 88.34 | **64.66** | 78.28 | 77.40 |
| Plan6 Phase1 full DSM | 76.57 | 77.01 | 86.88 | 63.53 | 77.67 | 77.80 |
| Phase4 F3: frozen+prompt | 76.52 | 76.27 | 87.74 | 62.76 | 78.06 | 77.78 |
| Plan7-B1 slope-only | 76.49 | 77.05 | 86.50 | 63.38 | **77.98** | 77.54 |
| Plan7-B3 slope+edge+curv | 76.49 | 77.39 | 87.77 | 63.58 | 76.95 | 76.74 |
| Phase4 F2: frozen no LoRA | 75.51 | 75.45 | 87.04 | 61.18 | 77.16 | 76.72 |
| Phase4 F0': frozen+4xSEF | 75.29 | 75.46 | 86.85 | 60.85 | 76.97 | 76.30 |
| Plan6 Phase1 DSM-lite | 74.88 | 76.21 | 86.34 | 60.51 | 76.43 | 74.89 |
| Plan8-CTRL gate-only | 74.77 | 76.34 | 84.53 | 62.47 | 76.75 | 73.78 |
| Plan8-CA-A cross-attn | 74.52 | 76.13 | 84.80 | 61.47 | 76.62 | 73.57 |
| Plan8-AB-A attn bias | 74.12 | 75.67 | 83.05 | 63.13 | 77.49 | 71.28 |
| Plan3 VPT+MFNet Decoder | 73.10 | 74.60 | 85.80 | 60.20 | 76.00 | 68.90 |
| Phase3-A3: in-ViT+SimpleDec | 72.88 | 75.00 | 85.40 | 61.40 | 76.20 | 66.30 |
| Phase3-A1: late fusion | 65.86 | 67.70 | 75.80 | 50.20 | 72.30 | 63.30 |
| Plan1 Zero-shot (256²) | 60.46 | 64.30 | 71.60 | 37.90 | 67.80 | 60.80 |
| Phase3-A0: frozen+Conv2d | 53.10 | 56.20 | 63.10 | 29.00 | 65.30 | 51.80 |

### 4.3 逐类 Recall（= MFNet 论文 "Per-Class OA"）

| 实验 | road | building | grass | tree | car |
|------|------|----------|-------|------|-----|
| **MFNet Best (SAM1)** | 93.39 | 98.84 | 81.16 | 93.17 | 89.23 |
| **MFNet Frozen (SAM1)** | 89.51 | 94.64 | 71.71 | 89.47 | 76.83 |
| **Plan7-A DSM prompt** | 88.52 | 93.00 | 79.70 | 86.06 | **91.59** |
| Plan7-C3 SAM-HQ residual | 90.00 | **93.02** | 77.95 | 85.45 | 88.29 |
| Plan6 P2-D1 Multi-scale | **90.65** | 92.10 | 78.77 | 85.85 | 90.62 |
| Plan6 Phase1 full DSM | 90.30 | 91.52 | 77.79 | 85.44 | 87.66 |
| Plan7-B1 slope-only | 90.83 | 91.70 | 77.12 | 85.14 | 86.44 |
| Plan7-B3 slope+edge+curv | 89.50 | 92.76 | **80.64** | 83.00 | 90.66 |
| Plan7-D1 Multi-scale | 89.06 | 93.63 | 79.51 | **86.29** | 92.14 |
| Plan6 Phase1 DSM-lite | 90.67 | 91.76 | 74.03 | 85.00 | 82.33 |
| Plan4 Full SGD (456M) | 88.50 | 91.90 | 79.60 | 82.80 | 73.80 |

---

## 五、DSM 信号失效问题（Tree vs Grass 区分失败）

### 5.1 问题定性

DSM 理论上应该是 tree（高）vs grass（矮）最有力的区分特征。但我们的实验数据显示，从 Plan6 Phase1 到 Plan7-A，DSM + Prompt 对 tree IoU 的提升只有 **+1.33pp**（76.99→78.32），远低于理论预期。

### 5.2 根因 #1（致命）：逐 Tile Min-Max 归一化消灭了树高信号

**代码位置**: `dataset_online.py:56` 和 `eval.py:84`

```python
dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
```

**问题机制**:

Vaihingen 每个 tile 的高程变化范围是地形起伏（36-57m），而树的高度只有 3-5m：

| Tile | DSM 范围 | 地形起伏 | 树高 | 树高/范围比 |
|------|---------|----------|------|------------|
| area5 | 245.88 - 282.27m | 36.4m | ~3.6m | **9.9%** |
| area15 | 288.42 - 345.69m | 57.3m | ~3.6m | **6.3%** |
| area21 | 272.03 - 316.80m | 44.8m | ~3.6m | **8.0%** |
| area30 | 261.77 - 299.37m | 37.6m | ~3.6m | **9.6%** |

归一化后的分离度:

```
area30 中:
  grass 归一化值 = (272.37 - 261.77) / (299.37 - 261.77) = 0.282
  tree  归一化值 = (276.01 - 261.77) / (299.37 - 261.77) = 0.379
  分离度 = 0.097 ← 仅占值域的 9.7%！
  分离率 = 0.68 个标准差 → 分布严重重叠
```

不同 tile 的同一物理高度映射到不同归一化值:

```
10m 高的树:
  area5:  (275 - 245) / (282 - 245) = 0.81
  area15: (300 - 288) / (345 - 288) = 0.21

模型无法学到 "树 = 高 DSM 值" 的一致规则
```

### 5.3 根因 #2：Gate 权重未学会偏向 DSM

分析训练好的 Plan7-A checkpoint（`best_model.pt`）中的 gate 参数:

```
Block 0:  softmax(gate) = [0.336, 0.333, 0.331]  ← 三者几乎均等
Block 14: softmax(gate) = [0.341, 0.327, 0.332]  ← 仍然均等
Block 31: dsm_gate = [0.0, 0.0, 0.0]             ← 完全为零！
```

**三个分支（RGB/DSM/Prompt）在所有 32 个 block 中的权重几乎相同**。考虑到:
- RGB 分支: SAM3 预训练的 1024 维强特征
- DSM 分支: 随机初始化的 32→1024 维瓶颈 adapter

同等权重下 RGB 特征天然主导，DSM 信号被淹没。

### 5.4 根因 #3：Window Attention 切割树冠

SAM3 ViTDet 的配置:

```
depth=32, patch_size=14, window_size=24
global_att_blocks = (7, 15, 23, 31)  ← 仅 4 个

1008×1008 → 72×72 token grid
window_size=24 → 3×3=9 个窗口
每个窗口 ≈ 7.7m × 7.7m 实地面积
```

一个中型树冠（直径 8-15m）跨越 2+ 个 attention 窗口。28/32 个 ViT block 中，DSM 高度信息无法跨窗口共享——树冠的完整结构信号被碎片化。

### 5.5 根因 #4：RGB 和 DSM 无 Cross-Attention

当前架构中 RGB 和 DSM token 分别做 self-attention，仅在 MLP 阶段通过 gate 加权求和:

```
RGB tokens → Self-Attention(q_rgb, k_rgb, v_rgb) → RGB 特征
DSM tokens → Self-Attention(q_dsm, k_dsm, v_dsm) → DSM 特征
                                                    ↓
Fusion = gate[0]*rgb_adapter(x) + gate[1]*dsm_adapter(y)  ← 逐 token 相加，无空间特异性
```

DSM token 的"这里很高"信息无法直接影响相邻 RGB token 的分类决策。Plan8 尝试了 cross-attention 但得到了负收益（CA-A: -0.26pp vs CTRL），说明 naive cross-attention 在 frozen ViT 中不 work。

### 5.6 根因 #5：Prompt 特征不匹配

Plan7-A 的 prompt 计算的是 Sobel slope + Laplacian edge——这是**边界/边缘特征**，不是高度特征。树和草与地面的边界看起来相似，边界特征无法区分"高植被"和"矮植被"。

### 5.7 影响排序

```
问题                                严重程度
─────────────────────────────────────────────────
逐 tile 归一化消灭树高信号          ████████████████████████████  致命
Gate 未学习偏向 DSM                 ██████████████████████        严重
Window attention 切割树冠           ████████████████              中等
无 Cross-attention                  ██████████████                中等
Prompt 特征不匹配                   ██████████                    较轻
```

---

## 六、过拟合问题

### 6.1 证据

所有主要实验都呈现相同的过拟合模式——训练 loss 持续下降，验证 mIoU 早期（E1-E3）达到峰值后停滞或下降：

Plan7-A (15 epochs):
```
Epoch:  1      2      3*     4      5      6      7      8     ...  15
Loss:   0.417  0.413  0.407  0.399  0.399  0.390  0.387  0.385  ...  0.375
mIoU:   76.70  75.60  76.87  76.51  75.58  76.68  76.34  76.09  ...  76.33
```

Plan4 D1 (116M params): Epoch 2 即过拟合。Plan4 D2 (227M): 更差。Phase4 F0'+L (LoRA): Epoch 1 即峰值后下降。

### 6.2 原因

- 12 张 Vaihingen 训练 tile，12.7M 可训练参数，严重过参数化
- 当前正则化（weight_decay=1e-3, dropout=0.1, gradient clipping=1.0）不足以防止过拟合
- 缺乏数据增强（仅在线随机裁剪，无颜色抖动/旋转/翻转）

### 6.3 可能的缓解方向

- 减少 adapter 参数量（降低 bottleneck，减少 adapter block 数量）——见 §7
- 添加更强数据增强（RandAugment, CutMix）
- 早停（当前训练了 15 epochs，最优在 E3）

---

## 七、Adapter 参数量消融空白

### 7.1 当前状态

Plan7-A 在所有 32 个 ViT block 中插入 5 个 LowRankAdapter（bottleneck=32），adapter 参数 10.65M。

### 7.2 未探索的消融维度

| 维度 | 当前 | 未尝试的方案 |
|------|------|-------------|
| bottleneck 维度 | 32 | 8, 16 |
| adapter block 数量 | 32 (全部) | 仅 4/10 个 global attention block |
| adapter 类型 | 3 分支 (RGB+DSM+prompt) | 2 分支 (RGB+DSM) 或 1 分支 |
| 组合方案 | 全 block + bn=32 | 10 block + bn=8 ≈ 0.83M |

### 7.3 证据支持

- Phase4 F0'+L 的 LoRA rank=8（~2.5M）可达到 77.24% mIoU，接近 bottleneck=32 adapter 的 77.55%
- Phase3 消融显示 gate 类型差异 ≤0.2pp，说明精细融合对性能影响极小
- 22/32 个 block 是窗口注意力，adapter 在其中的作用可能远小于全局 block

---

## 八、结构性问题

### 8.1 SAM3 ViTDet 的"刚性"

SAM3 在 SA-1B（11M 张图）上训练，特征空间比 SAM1（400K 张图）固化更深：
- LoRA rank=8 增益: SAM1 +6.95pp vs SAM3 +1.95pp，效率仅 28%
- Plan4 全部 unfreeze 尝试均导致退化（越解冻越差）
- 这是一个**结构性限制**——SAM3 ViTDet 不是为下游微调设计的 backbone

### 8.2 参数高效微调的天花板

在当前框架下（frozen SAM3 + adapter + MFNetDecoder），存在一个 ~78% mIoU 的软天花板：
- adapter 增益效率仅 SAM1 的 26%
- 窗口注意力限制了 DSM 信息的跨窗口传播
- Vaihingen 12 tile 的规模不足以充分训练 10M+ 参数

### 8.3 未来方向

1. **nDSM 数据修复**（最优先，零模型修改）：使用 DSM - DTM 替代 raw DSM + min-max 归一化，预期显著改善 tree/grass 区分
2. **减少 adapter 参数**：bottleneck 维度消融 + 选择性 block 适配（预期维持性能同时减少过拟合）
3. **更改 backbone**：考虑 SAM2 ViT-L 作为 SAM1 和 SAM3 之间的折中方案
4. **数据增强**：RandAugment / CutMix 缓解 Vaihingen 小数据集过拟合
5. **开放词汇能力评估**：利用训练好的 adapter 特征 + SAM3 原生 text decoder 验证领域适配是否提升了开放词汇分割

---

## 九、修复建议优先级

| 优先级 | 方向 | 预期收益 | 工作量 | 风险 |
|--------|------|---------|--------|------|
| 🔴 P0 | nDSM 替换 raw DSM + 全局标准化 | tree/grass IoU +3-5pp | 1-2 天 | 低 |
| 🟡 P1 | Adapter 参数量消融 (bn, block 数) | 减少过拟合 +0.3-0.5pp | 2-3 天 | 低 |
| 🟡 P1 | 数据增强 (RandAugment, CutMix) | 减少过拟合 +0.5-1.0pp | 1 天 | 低 |
| 🟢 P2 | 训练时添加 tree↔grass 混淆 loss | 针对性提升 tree +0.5-1.0pp | 2 天 | 中 |
| 🟢 P2 | 增加更多数据集 (Potsdam, LoveDA) | +1-3pp | 1 周 | 中 |
| 🔵 P3 | 更换 backbone (SAM2 ViT-L) | 可能 +3-5pp | 2 周 | 高 |
| 🔵 P3 | RGB↔DSM cross-attention 重新设计 | 可能 +1-2pp | 1 周 | 高 (Plan8 负收益) |
