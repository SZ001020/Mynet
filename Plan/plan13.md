# Plan13: 多层级 ViT 特征 + 植被细粒度区分

> 日期: 2026-06-05 ~ 2026-06-07
> 状态: **P13-G 已完成, 待 from-scratch 验证** — P13-G5 strict veg texture 在 256² 协议下 +0.48pp (76.72→77.20%), 是 Plan13 唯一正向结果。但热启动增益存在"优化红利"疑虑, 需要 from-scratch 对照实验区分架构贡献与优化贡献。
> 目标: 修复 ViT 多层级特征被系统性丢弃的架构缺陷，结合 vegetation refinement head 解决 tree↔grass 混淆（占总误分 ~20%）

---

## 1. 动机

### 1.1 Plan11 结论回顾

Plan11 的核心诊断: tree↔grass 混淆（veg sum ~20%）是 **RGB 纹理问题**，不是高程问题。

| 实验 | mIoU | grass→tree | tree→grass | veg sum |
|------|:---:|:---:|:---:|:---:|
| Plan7-A (soft-logit) | 77.55 | 12.3% | 7.5% | 19.8% |
| P11-B (nDSM, per-patch) | 76.72 | 10.3% | 9.2% | 19.6% |

nDSM 帮助了 road/car（绝对高度信息），但 tree↔grass 混淆几乎不变（19.8% → 19.6%，仅 -0.2pp）。

### 1.2 架构缺陷探查（2026-06-05）

当前架构存在**确定性的信息浪费**——SAM3 ViT 的 32 个 block 形成层级化特征表示，但只有最后 1 个 block（block 31）的特征被使用:

```
ViT block 0-6  (窗口注意力, 局部纹理)  → 丢弃
ViT block 7   (全局注意力, 纹理+上下文) → 丢弃  ← 包含纹理信息!
ViT block 8-14 (窗口注意力)             → 丢弃
ViT block 15  (全局注意力)              → 丢弃
ViT block 16-22                        → 丢弃
ViT block 23  (全局注意力)              → 丢弃
ViT block 24-30                        → 丢弃
ViT block 31  (全局注意力, 纯语义)      → 使用 → Neck FPN → Pyramid4Scale → Decoder
```

**关键参数**: `patch_size=14`, `global_att_blocks=(7,15,23,31)`, `return_interm_layers=False`

Neck 从 block-31 特征通过 ConvTranspose 人工"放大"出 1/4、1/8 分辨率特征，但**上采样无法恢复已丢失的纹理细节**。此外 model.py 还丢弃了 neck 已生成的 1/4、1/8 特征，用 Pyramid4Scale 重新生成——形成了双重浪费。

**设置 `return_interm_layers=True`** 即可获得 4 个不同语义深度的特征（block 7/15/23/31），改动向后兼容（neck 仍用 `xs[-1]`，行为不变）。

---

## 2. 实验设计总览

```
P13-A: 多层级 ViT 特征金字塔        ← 修复架构缺陷, 零新增参数 (仅 4 个 1×1 conv)
  ↓
P13-B: Vegetation refinement head   ← 专用 tree/grass 二阶段判别
  ↓
P13-C: RGB texture branch           ← 外部纹理补充 (若 ViT 内部纹理不够)
  ↓
P13-D: Gated nDSM auxiliary         ← nDSM 受控辅助
```

**排序逻辑**: 先改善 decoder **输入**质量（P13-A），再改善 decoder **输出**架构（P13-B），最后补充外部特征（P13-C）和 DSM 精细化（P13-D）。每阶段收益可独立归因。

---

## 3. P13-A: 多层级 ViT 特征金字塔

### 3.1 设计

将 4 个 ViT 全局注意力 block 的输出分别注入 decoder 的 4 个尺度:

```
ViT block 7  (纹理丰富)  → proj7  → upsample 4x  → 1/4  (288²) → decoder res1
ViT block 15              → proj15 → upsample 2x  → 1/8  (144²) → decoder res2
ViT block 23              → proj23 → identity     → 1/16 (72²)  → decoder res3
ViT block 31 (语义最强)   → proj31 → maxpool 2x   → 1/32 (36²)  → decoder res4
```

每个 `proj{N}` 是 `nn.Conv2d(1024, 256, 1)`，共 4×1024×256 ≈ **1.05M 新增参数**。

### 3.2 实现方式

- 在 `model.__init__` 中设置 `self.backbone.vision_backbone.trunk.return_interm_layers = True`
- 注册 forward hook 捕获 ViT trunk 的完整输出列表（`[feat_7, feat_15, feat_23, feat_31]`）
- 替换 `Pyramid4Scale` 为多源特征金字塔构建
- 不修改 `mm_adapter_vit.py`、`mfnet_decoder.py`、neck

### 3.3 关键实现细节

- ViT 输出为 BCHW 格式 `[B, 1024, 72, 72]`（patch_size=14, 1008/14=72）
- Block 7/15/23 的特征经过了对应深度的 RGB/DSM/prompt 融合（MMAdapterPromptBlock 在每个 block 内融合）
- `return_interm_layers=True` 向后兼容: neck 仍取 `xs[-1]`，行为不变
- activation checkpointing 兼容: 每个 block 独立 checkpoint，block 间输出可正常捕获
- 额外显存: ~60MB（3 个额外 72×72×1024 特征图，bf16）

### 3.4 训练配置

| 参数 | 值 |
|------|-----|
| 初始化 | 从头训练 (seed=42) |
| 基线对比 | P11-B (76.72%, 同 seed, 同 nDSM, 同架构仅无多层级特征) |
| dataset | Vaihingen, nDSM 预处理 |
| epochs | 15 |
| batch | 2 |
| epoch_steps | 1000 |
| resolution | 1008 |
| lr (adapter/decoder) | 5e-5 |
| lr (dsm/prompt encoder) | 2.5e-5 |
| lr (vit projections) | 5e-5 |
| loss | structure_loss (原始, 无 veg weight) |
| lineage_type | formal_ablation |
| comparison_role | multi-level ViT feature effect; baseline is P11-B |

### 3.5 验收标准

- 256² 滑动窗口 eval (soft-logit)
- 对比 P11-B 的 mIoU、OA、per-class recall
- **关键观察**: grass recall、tree recall、grass→tree、tree→grass
- 额外观察: road/car recall（多层级特征应普遍提升，不限于植被）
- 若 mIoU 低于 P11-B: 说明 ViT 内部纹理不足以区分植被，确认需要外部 texture branch（P13-C）

---

## 4. P13-B: Vegetation Refinement Head

### 4.1 设计

在 P13-A 的多层级特征基础上，增加专门的 tree/grass 二阶段判别:

```
Stage 1: P13-A 模型 → 6 类 logits + decoder features
                           ↓
veg_mask = (pred_grass > thresh) | (pred_tree > thresh)
                           ↓
Stage 2: Vegetation refinement head
  输入: [decoder_feat_1/4, decoder_feat_1/8] (P13-A 多层级特征)
  输出: tree/grass delta logits (residual 形式)
  训练 loss: 只在 GT vegetation 区域计算 CE
                           ↓
最终 logits: 非植被类保持 Stage 1, 植被类 = Stage 1 + delta
```

### 4.2 结构

```python
class VegetationRefinementHead(nn.Module):
    def __init__(self, in_channels=256):
        # 轻量: 2 层 conv + 输出
        self.fusion = nn.Sequential(
            ConvBNReLU(in_channels * 2, 128),  # 1/4 + 1/8 特征拼接
            nn.Conv2d(128, 2, kernel_size=1),    # tree/grass delta logits
        )

    def forward(self, feat_1_4, feat_1_8, veg_mask):
        # feat_1_4: [B, 256, H/4, W/4]
        # feat_1_8: [B, 256, H/8, W/8] → upsample to H/4
        feat_1_8_up = F.interpolate(feat_1_8, scale_factor=2)
        fused = torch.cat([feat_1_4, feat_1_8_up], dim=1)
        delta = self.fusion(fused)  # [B, 2, H/4, W/4]
        return delta * veg_mask.unsqueeze(1).float()  # 只作用于植被区域
```

### 4.3 训练配置

| 参数 | 值 |
|------|-----|
| 初始化 | P13-A 最佳 checkpoint |
| 新增参数 | ~0.15M |
| lr (refinement head) | 5e-5 |
| lr (backbone 其余) | 1e-5 (微调) |
| loss | CE(veg_delta + stage1_logits, gt), 仅 veg 区域计算 |
| 评估模式 | oracle veg mask + pred veg mask 双报告 |
| lineage_type | continuation |
| comparison_role | vegetation refinement head effect; baseline is P13-A |

### 4.4 验收标准

- oracle veg mask 下 grass→tree、tree→grass 是否显著下降
- pred veg mask 下指标是否接近 oracle
- oracle/pred gap 判断瓶颈在 Stage 1 mask 还是 refinement head
- overall mIoU 不低于 P13-A

---

## 5. P13-C: RGB Texture Branch

### 5.1 设计

若 P13-A 和 P13-B 已解决大部分植被混淆，跳过此阶段。若仍有显著 veg sum，说明 ViT 内部纹理不足以区分 tree/grass 局部纹理差异，需要外部 RGB 纹理分支。

### 5.2 结构

```python
class TextureStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # H/2
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # H/4
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # H/4
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # H/4
        )

    def forward(self, rgb):
        return self.stem(rgb)  # [B, 256, H/4, W/4]
```

融合方式: texture feature 与 P13-A 的 decoder 1/4 特征拼接，输入 P13-B refinement head。

### 5.3 训练配置

| 参数 | 值 |
|------|-----|
| 初始化 | P13-B 最佳 checkpoint |
| 新增参数 | ~0.3M |
| loss | 只在 vegetation 区域计算 CE |
| lineage_type | continuation |
| comparison_role | external RGB texture effect; baseline is P13-B |

---

## 6. 最终实验结果

### 6.1 总览 — 两套协议

| 实验 | 256² mIoU | Δ vs P11-B | MFNet mIoU | Δ vs P11-B |
|------|:---:|:---:|:---:|:---:|
| P11-B (基线) | 76.72% | — | 85.31% | — |
| P13-A (multi-level ViT) | 76.42% | -0.30 | 84.84% | -0.47 |
| P13-C (texture branch + gate) | 76.70% | -0.02 | 85.55% | +0.24 |
| P13-E (building gate) | 76.54% | -0.18 | 85.53% | +0.22 |
| P13-F (per-class gate) | 76.64% | -0.08 | 85.61% | +0.30 |
| **P13-G5 (strict veg texture)** | **77.20%** | **+0.48** | **85.30%** | **-0.01** |

### 6.1.1 P13-G 五组消融结果

训练中 crop val (1008²), 所有变体从 P11-B 热启动:

| 实验 | Mask | 特征 | Loss | oracle mIoU | pred mIoU |
|------|:--:|------|:--:|:---:|:---:|
| G1 | oracle | RGB multiscale | CE | 78.60% | 78.79% |
| G2 | pred | RGB multiscale | CE | 78.57% | 78.78% |
| G3 | oracle | RGB + nDSM roughness | CE | 78.69% | 78.83% |
| G4 | pred | RGB + nDSM roughness | CE | 78.68% | 78.83% |
| **G5** | pred | RGB + nDSM roughness | **DiceCE** | 78.89% | **78.90%** |

**256² 滑动窗口最终评估 (G5)**:

| 协议 | mIoU | OA | road | building | grass | tree | car | veg sum |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 256² stride=128 | **77.20%** | 87.74% | 77.10 | 88.21 | 64.26 | 78.35 | 78.09 | 776,279 |
| MFNet exact | **85.30%** | 92.30% | 86.24 | 93.24 | 71.42 | 86.12 | 89.51 | — |

### 6.2 MFNet 标准协议完整对比

| 实验 | mIoU | OA | roads | buildings | grass | trees | cars | g→t | t→g |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P11-B | 85.31 | 92.28 | 94.3 | 95.3 | 82.6 | 92.5 | 98.1 | 9.7% | 6.4% |
| P13-A | 84.84 | 92.05 | 94.0 | 94.9 | 82.6 | 92.4 | 98.2 | 10.0% | 6.6% |
| P13-C | 85.55 | 92.35 | 94.0 | 95.5 | 82.4 | 93.1 | 97.9 | 9.9% | 5.8% |
| P13-E | 85.53 | 92.28 | 94.5 | 95.1 | 82.8 | 92.4 | 97.9 | 8.8% | 6.4% |
| **P13-F** | **85.61** | **92.33** | **93.4** | **95.5** | **82.2** | **93.8** | **97.6** | **10.9%** | **5.2%** |

per-class 列为 Recall (TP/(TP+FN))

### 6.3 各阶段 TP 变化 (vs P11-B, MFNet 协议)

| 实验 | roads | buildings | grass | trees | cars | 模式 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| P13-A | -0.32pp | +0.21pp | -0.25pp | +0.54pp | -0.20pp | 噪声 |
| P13-C | -0.32pp | +0.21pp | -0.25pp | +0.54pp | -0.20pp | tree↑ road↓ |
| P13-E | +0.16pp | -0.14pp | +0.23pp | -0.17pp | -0.21pp | road↑ tree↓ |
| P13-F | -0.95pp | +0.21pp | -0.42pp | +1.26pp | -0.51pp | tree↑↑ road↓↓ |

### 6.4 混淆矩阵共识分析

三个纹理变体 (P13-C/E/F) 交叉分析:

- **一致改善的误分流向**: 无 (没有任何流向被三个变体一致减少)
- **一致退化的误分流向**: low veg. → buildings (三个变体全部增加此误分)
- **方向分歧**: P13-C/F 推 tree 吃 road, P13-E 推 road 吃 tree

### 6.5 最终结论

1. **P13-A (多层级 ViT)**: 两套协议下均负向 (-0.30/-0.47pp)。ViT 14×14 patch 是根本瓶颈, 内部纹理信息不足以区分类别。**关闭。**

2. **P13-C/E/F (全图 texture delta)**: MFNet 协议下 +0.22~+0.30pp, 旧协议下 -0.02~-0.18pp。全部在 ±0.3pp 噪声范围内。纹理分支学到了树冠纹理特征 (tree recall 最高 +1.26pp), 但代价是 road recall 下降。**本质是全图类别 logits 的零和重排序, 不是净提升。关闭。**

3. **核心认知修正**: P13-C/E/F 不能证明"纹理方向整体不行", 只能证明当前实现的**全图 texture delta**会扰动 road/building/car 与 grass/tree 的类别竞争。真正的 vegetation-only 纹理判别仍需单独验证。

### 6.6 全部加入失败方向表

| 方向 | 证据 | Δ | Plan |
|------|------|:---:|:---:|
| 多层级 ViT 特征金字塔 (block7/15/23/31) | P13-A 76.42 vs P11-B 76.72 (旧) / 84.84 vs 85.31 (MFNet) | -0.30 / -0.47 | Plan13 |
| 外部 RGB 纹理 CNN + 全图 logits refinement | P13-C/E/F 均 ≤0.3pp, 无共识改善, 且扰动 road/building/car | 噪声级 | Plan13 |

---

## 7. P13-D: Gated nDSM Auxiliary（未执行）

> P13-A 和 P13-C 均无效，按决策树不进入此阶段。

### 7.1 设计

nDSM 不应被丢弃（P11-B 证明对 road/car 有效），但不应作为 tree/grass 的硬判据。使用 gate 机制让模型选择性信任 nDSM:

```python
# 在 refinement head 中:
gate = torch.sigmoid(self.gate_conv(torch.cat([semantic_feat, texture_feat, ndsm_feat], dim=1)))
fused = semantic_feat + gate * ndsm_feat
```

### 7.2 诊断需求

- 保存 gate heatmap: 观察 tree/grass 边界处 gate 是否降低
- 统计 gate 分布: 检查是否退化为常数（全 0 或全 1）

### 7.3 训练配置

| 参数 | 值 |
|------|-----|
| 初始化 | P13-C 最佳 checkpoint |
| lineage_type | continuation |
| comparison_role | gated nDSM effect; baseline is P13-C |

---

## 8. P13-G: 严格 Vegetation-only 多尺度纹理修正版（已完成）

> P13-C/E/F 的实现并未真正限制在 vegetation 区域内。`veg_delta` 在 forward 中被直接加到全图 grass/tree logits, 训练主 loss 也对全图 refined logits 生效, 因而实验测到的是"全图类别竞争重排", 不是严格的 tree/grass 内部细粒度判别。

### 8.1 文献依据

相关高质量结论:

1. **MultiVeg**: Tree 与 Low Vegetation 的标注依据是影像中可见的 texture 与 spatial context, 而不是绝对高度或物种。Tree 表现为粗糙、不规则、树冠/阴影模式明显; Low Vegetation 更平滑、均质。
2. **RGB-only Tree/Shrub/Grass**: 将分类限制在 vegetation classes 内部时, tree/grass/shrub 准确率高于与全部 land-cover 类联合训练。
3. **CLBP urban vegetation**: 不同植被类型对应不同最优纹理窗口。grass/shrub 约 `3×3`, arbor/tree 约 `11×11`, 混合植被约 `7×7`/`9×9`。
4. **Vaihingen 2D/3D-GLCM**: 在 spectral+geometry 基础上加入 2D/3D texture 可提升 low vegetation 和 tree F1; 3D-GLCM 优于 2D-GLCM, 说明 nDSM 局部结构可作为纹理/粗糙度使用。
5. **Morphological granulometry**: 多尺度 opening/closing 可描述地物颗粒大小, 适合建模树冠粗糙度与草地平滑度差异。

### 8.2 与 P13-C/F 的关键区别

| 项目 | P13-C/F 已做 | P13-G 要求 |
|------|--------------|------------|
| delta 作用范围 | 全图 grass/tree logits | 仅 vegetation mask 内 |
| 非植被类 | 会被 grass/tree delta 间接扰动 | 完全保持 P11-B base logits |
| 训练目标 | `structure_loss(refined, labels)` 全图生效 | 主模型冻结, 只训练 vegetation binary head |
| 纹理建模 | 单一路径 CNN texture stem | 显式多尺度纹理: `3×3/7×7/11×11` |
| nDSM 用法 | 单值高度/adapter token | roughness / local variance / slope / morphology residual |
| 验证重点 | mIoU/OA | vegetation-only accuracy + `grass→tree/tree→grass/veg sum` |

### 8.3 推荐结构

```text
P11-B frozen
  ├─ base_logits: [road, building, grass, tree, car]
  ├─ base_feats: decoder 1/4 feature 或 logits-side feature
  └─ pred_veg_mask = argmax(base_logits) in {grass, tree}

RGB
  ├─ 3×3 texture branch   # grass fine texture
  ├─ 7×7 texture branch   # mixed vegetation
  └─ 11×11 texture branch # tree/arbor coarse texture

nDSM
  ├─ local variance
  ├─ max_pool - min_pool roughness
  ├─ slope magnitude
  └─ opening/closing residual

[base_feats, RGB multiscale texture, optional nDSM roughness]
  ↓
vegetation binary head
  ↓
tree/grass logits or delta logits
```

最终合成必须严格满足:

```python
final_logits = base_logits.clone()

# 只在 vegetation mask 内更新 grass/tree
veg_mask_2ch = veg_mask.unsqueeze(1).expand_as(final_logits[:, 2:4])
final_logits[:, 2:4] = torch.where(
    veg_mask_2ch,
    refine_grass_tree_logits,
    base_logits[:, 2:4],
)

# 非 vegetation 区域完全等于 P11-B
nonveg_mask = ~veg_mask.unsqueeze(1).expand_as(final_logits)
assert torch.equal(final_logits[nonveg_mask], base_logits[nonveg_mask])
```

优先使用 residual 形式:

```text
refine_grass_tree_logits = base_logits[:, 2:4] + delta_grass_tree
```

### 8.4 两阶段验证协议

必须同时报告 oracle 与 pred 两种 mask:

| 模式 | vegetation mask | 目的 |
|------|-----------------|------|
| P13-G-oracle | GT grass/tree mask | 验证 texture binary head 的理论上限 |
| P13-G-pred | P11-B pred grass/tree mask | 验证真实 pipeline 收益 |

判断逻辑:

- oracle 有效, pred 无效: Stage-1 vegetation mask 是瓶颈。
- oracle 无效: 当前 RGB+nDSM texture 信息不足, 应关闭纹理方向。
- oracle/pred 都有效: vegetation-only 多尺度纹理方向成立。

### 8.5 训练配置

| 参数 | 值 |
|------|-----|
| 初始化 | P11-B checkpoint |
| P11-B 主模型 | frozen |
| 新增模块 | `MultiScaleTextureStem`, `NDSMRoughnessStem`, `VegetationBinaryHead` |
| loss | CE / DiceCE, 仅 GT vegetation 区域 |
| batch | 2 |
| epochs | 10 |
| lr | 5e-5 (new modules only) |
| comparison_role | strict vegetation-only multiscale texture effect; baseline is P11-B |

五个执行变体:

| 实验 | mask 选择 | 特征 | loss | best 选择 |
|------|-----------|------|------|-----------|
| P13-G1 | oracle | RGB `3×3/7×7/11×11` | CE | oracle mIoU |
| P13-G2 | pred | RGB `3×3/7×7/11×11` | CE | pred mIoU |
| P13-G3 | oracle | RGB 多尺度 + nDSM roughness | CE | oracle mIoU |
| P13-G4 | pred | RGB 多尺度 + nDSM roughness | CE | pred mIoU |
| P13-G5 | pred | RGB 多尺度 + nDSM roughness | DiceCE | pred mIoU |

训练 loss:

```python
veg_gt = (labels == 2) | (labels == 3)
veg_logits = final_logits[:, 2:4]
loss = F.cross_entropy(veg_logits.permute(0,2,3,1)[veg_gt], labels[veg_gt] - 2)
```

注意: 不对非 vegetation pixels 计算 refinement loss, 不允许 texture head 通过 full-image `structure_loss` 改变 road/building/car。

### 8.6 实际结果

**核心发现 (256² stride=128)**:

| 指标 | P11-B | P13-G5 | Δ |
|------|:---:|:---:|:---:|
| mIoU | 76.72% | 77.20% | **+0.48** |
| OA | 87.24% | 87.74% | +0.50 |
| grass IoU | 63.70% | 64.26% | +0.56 |
| tree IoU | 78.06% | 78.35% | +0.29 |
| building IoU | 86.47% | 88.21% | **+1.74** |
| veg sum | 797,744 | 776,279 | **-21,465 (-2.7%)** |

**关键观察**:

1. **Oracle vs pred mask 几乎无差异** (~0.03pp): P11-B 的 vegetation mask 已经足够准, 不需要 GT
2. **nDSM roughness 加成微弱**: G4 vs G2 = +0.05pp — 噪声级
3. **DiceCE loss 微弱正向**: G5 vs G4 = +0.07pp
4. **building +1.74pp 是真实改善**: P11-B 错分为 grass/tree 的 building 边缘像素, 被 refinement head 压低 veg logit 后纠正回 building。机理是 strict mask 保证非植被 logits 不变, 压低植被 logit 后 argmax 翻转到 building
5. **MFNet 协议排名反转 (85.30% vs P13-F 85.61%)**: eroded labels 排除了建筑-植被边界像素, 正好是 P13-G5 主要改善的区域。P13-C/E/F 的 gate 机制改善内部像素, 在 MFNet 下保留; P13-G5 改善边界, 在 MFNet 下被排除
6. **tree↔grass 不是单调双向改善**: 256² 下 grass→tree 从 346,964 增至 429,430, tree→grass 从 450,780 降至 346,849, 双向合计下降。G5 更像是把 tree/grass 方向性 trade-off 重新平衡, 同时修复一部分 building 边界。

**结论**: strict vegetation mask + 多尺度纹理方向**在 256² 协议下是 Plan13 唯一正向信号 (+0.48pp)**。但热启动无法区分架构贡献与优化贡献。MFNet 协议下无增益说明改善集中于边界。需 from-scratch 验证。

失败判据回应:

- P13-G-oracle / pred 的 crop-val 统计均 `invariant_errors=0`, 证明 mask 外 logits 未被改动。
- P13-G-pred 有效且 mIoU 上升: ✅ 256² +0.48pp, `veg_sum` -21,465 (-2.7%)
- 但 MFNet 协议下无改善: 改善集中在边界 → 需 from-scratch 验证

---

## 9. P13-G from-scratch 验证实验（待执行）

> 新增原因: P13-G5 的 +0.48pp 来自 P11-B 热启动, 无法区分是架构改进还是优化红利（更多训练步数、更好初始化）。唯一验证方法是 from-scratch 训练。

### 9.1 核心问题

热启动的提升有三种可能来源:

| 来源 | 解释 | 是否可复现 | 状态 |
|------|------|:---:|:---:|
| A: 架构改进 | 多尺度纹理 + strict veg mask 提供了新信息通路 | ✅ 可复现 | 待验证 |
| B: 优化红利 | P11-B 欠收敛, 热启动 = 变相多训 10 epoch | ❌ 不可复现 | **已排除** |
| C: 初始化红利 | 好的起点 → 更好的局部最优 | ❌ 不可复现 | 待验证 |

**P11-B 50 epoch 续训实验已排除类型 B**:

| 实验 | 配置 | 结果 |
|------|------|------|
| P11-B (原始) | 15 epoch, seed=42 | 76.72% (256²), crop val ~76.71% |
| P11-B 50 epoch | 50 epoch, seed=42, 同架构 | crop val best 76.89%, epoch 40-49 完全水平 (76.28-76.40%) |

P11-B 从 epoch 15 到 50 基本不涨（crop val +0.18pp），证明 **P11-B 已收敛**。P13-G5 的 +0.48pp 增益不是"多训练了几步"。剩余问题是：这 +0.48pp 是架构贡献（A）还是初始化红利（C）。唯一验证方法仍然是 from-scratch 训练。

### 9.2 实验设计

**单一实验**: P13-G from-scratch, 与 P11-B from-scratch (同 seed=42) 直接对比。

| 参数 | 值 |
|------|-----|
| 实验名 | P13-G-scratch |
| 架构 | P11-B (Plan7-A with nDSM) + P13-G strict veg head (G5 配置: RGB multiscale + nDSM rough + DiceCE) |
| 初始化 | **从头训练 (seed=42)**, 不加载 P11-B checkpoint |
| 基线 | P11-B (76.72%, 同 seed=42, 同 nDSM, 同 15 epoch) |
| dataset | Vaihingen |
| epochs | 15 |
| batch | 2 |
| epoch_steps | 1000 |
| resolution | 1008 |
| lr (adapter/decoder) | 5e-5 |
| lr (veg head) | 5e-5 |
| lr (dsm/prompt encoder) | 2.5e-5 |
| loss | structure_loss (主 decoder) + DiceCE (veg head, 仅 veg 区域) |
| lineage_type | formal_ablation |
| comparison_role | strict veg texture architectural effect vs pure P11-B; both from scratch seed=42 |

### 9.3 预期结果与判断

| 结果 | 判断 |
|------|------|
| P13-G-scratch > P11-B (Δ > +0.3pp) | 架构有效, strict veg texture 是真正的改善, 可进入 Plan14 正式链路 |
| P13-G-scratch ≈ P11-B (Δ ≈ 0) | 热启动红利, 架构本身无贡献, 关闭方向 |
| P13-G-scratch < P11-B | veg head 干扰了 from-scratch 训练, 方向关闭 |

### 9.4 P11-B 续训结果（已完成）

| 实验 | 路径 | 配置 | 结果 |
|------|------|------|------|
| P11-B 50 epoch | `plan11_b_ndsm_vaihingen_20260607_004826` | seed=42, 同架构, checkpoint_attn=false | crop val best 76.89%, epoch 15-50 完全平台 |

**结论**: P11-B 已收敛, P13-G5 的 +0.48pp 不是优化红利。from-scratch 验证只需区分"架构改进"vs"初始化红利"。

---

## 10. 代码结构

```
Personal-Project/RS-SAM3-p13/
├── phase_a_multilevel_vit/        # P13-A: 多层级 ViT 特征金字塔
│   ├── model.py                   #   修改自 RS-SAM3-p7/phase_a_dsm_prompt/model.py
│   ├── mm_adapter_vit.py          #   复制自 RS-SAM3-p7 (不改动)
│   ├── mfnet_decoder.py           #   复制自 RS-SAM3-p7 (不改动)
│   ├── dsm_prompt.py              #   复制自 RS-SAM3-p7 (不改动)
│   ├── structure_loss.py          #   复制自 RS-SAM3-p7 (不改动)
│   ├── dataset_online.py          #   复制自 RS-SAM3-p11/phase_b_ndsm (nDSM)
│   ├── dataset_adapter.py         #   复制自 RS-SAM3-p7 (不改动)
│   ├── train.py                   #   修改自 RS-SAM3-p11/phase_b_ndsm/train.py
│   └── eval.py                    #   复制自 RS-SAM3-p11 (256² 滑动窗口)
├── phase_b_veg_refinement/        # P13-B: Vegetation refinement head
│   ├── model.py
│   ├── veg_refine_head.py         #   新增: refinement head
│   ├── train.py
│   └── eval.py
├── phase_c_texture_branch/        # P13-C: RGB texture branch
│   ├── model.py
│   ├── texture_stem.py            #   新增: texture CNN
│   ├── train.py
│   └── eval.py
├── phase_d_gated_ndsm/            # P13-D: Gated nDSM auxiliary
│   ├── model.py
│   ├── train.py
│   └── eval.py
└── phase_g_strict_veg_texture/    # P13-G: strict vegetation-only multiscale texture
    ├── model.py                   #   P11-B frozen + texture/nDSM roughness binary head
    ├── texture_multiscale.py      #   3×3/7×7/11×11 RGB texture branches
    ├── ndsm_roughness.py          #   local variance/slope/morphology residual
    ├── train.py                   #   oracle GT vegetation mask training
    └── eval.py                    #   oracle + pred vegetation mask 双评估
```

---

## 11. 风险与缓解

| 风险 | 说明 | 缓解 |
|------|------|------|
| ViT 早期特征对植被纹理区分力不足 | Block 7 已过 7 层 transformer + patch_embed (14×14 patches)，纹理细节可能已不够 | 先做 P13-A 快速验证 (1 epoch 看趋势)；若无效则直接跳到 P13-C |
| forward hook + activation checkpointing 兼容性 | hook 捕获的中间特征可能缺少梯度 | 测试首个 epoch 的 loss backward 是否正常 |
| return_interm_layers 改变 ViT channel_list | `channel_list` 从 `[1024]` 变为 `[1024,1024,1024,1024]` | 不影响 neck（neck 用 `channel_list[-1]`，仍为 1024） |
| 新增 proj 层与旧 checkpoint 不兼容 | strict=False 可加载 adapter/decoder 权重 | P13-B 从 P13-A 加载时 strict=False |
| 显存增加 | 额外 3 个 72×72×1024 特征图 ≈ 60MB (bf16) | 在 32GB RTX 5090 上可忽略 |
| P13-G 仍退化为类别重排 | texture head 如果作用全图会复现 P13-C/F 问题 | 强制 mask 外 logits 等于 P11-B, 加 assert/单元检查 |
| GT mask 与 pred mask gap 大 | oracle 有效但真实 pipeline 无收益 | 单独报告 oracle/pred gap, 不把 pred 失败误判为 texture 失败 |
| 多尺度纹理过拟合 | Vaihingen 训练 tile 少 | P11-B frozen, 只训练轻量 head, 做 seed 检查 |

---

## 12. 执行顺序

```
Step 1: 创建目录结构, 复制基线代码
Step 2: P13-A 模型实现 + 冒烟测试 (1 epoch, 验证 loss 下降)
Step 3: P13-A 正式训练 (15 epochs) + 256² eval
Step 4: 根据 P13-A 结果决策:
        - 若 veg sum ↓ >2pp 且 mIoU ↑ → 继续 P13-B
        - 若 veg sum 不变但其他类提升 → 记录, 继续 P13-B
        - 若 mIoU ↓ → 跳至 P13-C
Step 5: P13-B 实现 + 训练 + eval
Step 6: 根据 P13-B 结果决策 P13-C/D
Step 7: 基于实现复盘与文献调研, 执行 P13-G:
        - G-oracle: GT vegetation mask
        - G-pred: P11-B predicted vegetation mask
        - mask 外 logits 必须与 P11-B 完全一致
```

---

## 13. 预期判断逻辑

### 情况 A: P13-A 显著降低 veg sum (>2pp)

多层级 ViT 特征是关键增量。后续重点: P13-B refinement head 精化 + P13-D nDSM gate。

### 情况 B: P13-A 仅提升非植被类

ViT 早期特征对纹理区分不够（14×14 patch 太粗糙），植被问题需要外部纹理分支。跳至 P13-C。

### 情况 C: P13-A 无收益

确认 14×14 patch 的 ViT 内部纹理信息不足以区分 tree/grass。P13-C（RGB texture branch，原始分辨率 CNN）成为主要方向。

### 情况 D: P13-B oracle 有效但 pred 无效

Stage 1 的 vegetation mask 质量是瓶颈，需强化 mask 预测而非 refinement head。

### 情况 E: P13-G oracle 有效且 pred 有效

严格 vegetation-only 多尺度纹理方向成立。后续可将 P13-G 移入 Plan14 正式链路, 并考虑 nDSM roughness/gated roughness 消融。

### 情况 F: P13-G oracle 有效但 pred 无效

纹理判别器有效, 但 Stage-1 vegetation mask 是瓶颈。后续不再加 texture, 先提升 tree∪grass mask 召回与精度。

### 情况 G: P13-G oracle 无效

当前 Vaihingen RGB+nDSM 中可用的 tree/grass 纹理信息不足, 可以更有力地关闭纹理方向。

---

## 14. 基线 Checkpoint

| 用途 | 路径 | 指标 |
|------|------|:---:|
| P13-A 基线对比 | P11-B: `/root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_20260604_225146/best_model.pt` | 76.72% |
| 代码参考 (nDSM) | `Personal-Project/RS-SAM3-p11/phase_b_ndsm/` | — |
| 模型架构参考 | `Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/` | 77.55% |
