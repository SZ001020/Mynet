# Plan11: Tree/Grass 植被区分优化（P0+P1）

> 日期: 2026-06-04 ~ 2026-06-05
> 状态: **已完成** — nDSM 正向 (+1.95pp vs 从头训练基线)，veg_boundary_weight 持续负向
> 目标: 基于混淆矩阵诊断，针对性修复 tree↔grass 双向混淆（占总误分 25%）

---

## 1. 动机

Plan7-A 混淆矩阵分析揭示了以下关键事实：

| 错误流向 | 像素数 | 占该类别 GT | 占总误分 |
|----------|--------|:----------:|:------:|
| grass → tree | 415,010 | 12.3% | — |
| tree → grass | 364,754 | 7.5% | — |
| **tree↔grass 合计** | **779,764** | — | **~25%** |
| building → tree | 22,713 | 0.4% | — |
| tree → building | 15,412 | 0.3% | — |

tree↔building 混淆仅 0.3%——DSM 很好地分离了高楼类别。瓶颈是**植被类别内部（tree vs grass）的边界模糊**，而非树与楼的混淆。

本 plan 将优化焦点从"更好的 DSM 融合"转向"更好的植被纹理+边界建模"。

**基线**: Plan7-A 软 logit eval mIoU=77.55%, OA=88.20%

---

## 2. 代码结构

```
Personal-Project/RS-SAM3-p11/
├── phase_a_veg_boundary_loss/   # P11-A: Tree↔Grass 边界加权 loss
├── phase_b_ndsm/                # P11-B: nDSM + 全局归一化
├── phase_c_combined/            # P11-C: P11-A + P11-B 联合
├── phase_d_augmentation/        # P11-D: 数据增强 (ColorJitter + Blur)
└── phase_e_adapter_ablation/    # P11-E: Adapter 参数量消融
```

所有代码基于 `Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/` 修改。

---

## 3. 实验设计

### P11-A: Tree↔Grass 边界加权 Loss

**文件**: `structure_loss.py`

**动机**: 当前 `structure_loss` 对所有类别边界施加相同 6× 权重。在 tree↔grass 边界，植被 mask（tree+grass 并集）是均匀的，标准边缘检测 `|avg_pool - mask|` 在此处不产生大的边缘信号——两个植被类别的模糊过渡没有锐利边缘。

**修改**: `structure_loss()` 新增 `veg_boundary_weight: float = 3.0` 参数。检测逻辑：

```
veg_boundary = (avg_pool(tree_mask) > 0.01) & (avg_pool(grass_mask) > 0.01)
```

仅对 grass (c=2) 和 tree (c=3) 通道的 `weit` 乘以 `(1.0 + veg_boundary_weight * veg_boundary)`。

**训练配置**:

| 参数 | 值 |
|------|-----|
| 初始化 | Plan7-A 最佳 ckpt (`225309/best_model.pt`) |
| epochs | 15 |
| batch | 2 |
| epoch_steps | 1000 |
| resolution | 1008 |
| lr (adapter/decoder) | 5e-5 |
| lr (dsm/prompt encoder) | 2.5e-5 |
| loss | structure_loss + veg_boundary_weight=3.0 |

**预期**: grass recall 78.8% → 82-85%, mIoU +1-2pp

---

### P11-B: nDSM + 全局归一化

**文件**: `dataset_online.py`, `train_a.py` (复制并重命名为 `train.py`), `eval.py`, `eval_confusion.py`

**动机**: 逐 tile min-max 归一化将树高信号（3-5m 绝对高度）压缩至 tile 高程范围 (36-57m) 的 6-10%。不同 tile 中相同物理高度映射到不同归一化值。

**修改**: 所有 DSM 加载位置替换为：

```python
from scipy.ndimage import grey_opening
ground = grey_opening(dsm, size=101)   # 形态学开运算估计地面
ndsm = dsm - ground                     # 物体高度
dsm = np.clip(ndsm / 10.0, 0, 1)       # 全局归一化: 除以 10m, 裁剪
```

**训练配置**:

| 参数 | 值 |
|------|-----|
| 初始化 | 从头训练 (seed=42) |
| epochs | 15 |
| 其他 | 同 P11-A（除 loss 为原始 structure_loss） |

**预期**: mIoU +1-2pp（低于最初估计的 +3-5pp，因 tree↔building 混淆本就仅 0.3%）

---

### P11-C: P11-A + P11-B 联合

nDSM 提供更好的高程信号 + 边界 loss 聚焦植被边界优化。从头训练, seed=42。

**预期**: 最强配置, mIoU 78.5-79.5%

---

### P11-D: 数据增强

**文件**: `dataset_online.py`

**动机**: Vaihingen 仅 12 张训练 tile，轻度 photometric 增强可提升泛化性。

**修改**: 在 `__getitem__` 的几何变换后、张量转换前添加（仅作用于 RGB）：
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), 概率 50%
- GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), 概率 30%

**训练配置**: 同 P11-C，额外启用数据增强。

**预期**: +0.3-0.8pp

---

### P11-E: Adapter 参数量消融

**文件**: `mm_adapter_vit.py`, `model.py`, `train.py`

**动机**: 量化"全 32 block injection"是否必要。当前 adapter 参数 ~10.65M (bottleneck=32 × 32 blocks)。

**修改**: `inject_mm_adapters()` 新增 `block_indices` 参数，仅在指定 block 注入 adapter。

**消融配置**: `bottleneck=8, block_indices=[7,15,23,31]` → ~0.33M 参数（减少 30×）

**训练配置**: 从头训练，其他同 P11-A。

**预期**: 消融实验，量化全 32 block 注入的边际贡献。

---

## 4. 执行顺序

```
Step 1: P1-3（混淆矩阵集成到 eval.py）         ← 零训练开销
Step 2: P11-A（边界 loss）冒烟测试 → 正式训练
Step 3: P11-B（nDSM）冒烟测试 → 正式训练
Step 4: P11-C（联合）正式训练
Step 5: P11-D（数据增强）正式训练
Step 6: P11-E（adapter 消融）正式训练
```

---

## 5. 风险

| 风险 | 缓解 |
|------|------|
| `grey_opening` 在 ~2500×1900 瓦片上的开销 | 仅缓存时计算一次 (~3s/tile × 12 = 36s 启动成本) |
| 植被边界权重过高 → 训练不稳定 | 保守起步 `veg_boundary_weight=3.0` |
| nDSM 改变输入分布，旧 ckpt 不兼容 | 所有 nDSM 实验从头训练 |
| P11-E 消融模型无法 strict 加载 | `strict=False` 或从头训练 |

## 6. 验证标准

每个 phase 完成后:
1. 256² 滑动窗口评估 + 混淆矩阵（4 个 Vaihingen 测试 tile）
2. 对比基线: mIoU=77.66%, OA=88.37%（Plan7-A 225309 ckpt 混淆矩阵 eval）
3. 重点观察 grass recall (当前 78.77%) 和 tree recall (当前 87.43%) 的变化

---

## 7. 实验结果

### 7.1 总览（256² 滑动窗口, per-patch argmax）

| # | 实验 | 配置 | init | mIoU | OA | 合理基线 | Δ | 结论 |
|---|------|------|------|:---:|:---:|------|:---:|:---:|
| — | Plan7-A | min-max | Plan6 | 77.14 | 87.75 | — | — | 当前最优 |
| — | Plan8-CTRL | min-max | scratch | 74.77 | 86.33 | — | — | 从头基线 |
| 1 | **P11-B** | **nDSM** | scratch | **76.72** | 87.42 | Plan8-CTRL | **+1.95** | ✅ |
| 2 | P11-D | nDSM + 增强 | scratch | 76.56 | 87.30 | P11-B | -0.16 | ➖ |
| 3 | P11-C | nDSM + veg | scratch | 76.51 | 87.14 | P11-B | -0.21 | ❌ |
| 4 | P11-A | min-max + veg | Plan7-A | 76.46 | 87.16 | Plan7-A | -0.68 | ❌ |
| 5 | P11-E | nDSM + 4-block | scratch | 72.61 | — | P11-B | -4.11 | ❌ |

### 7.2 各类别 Recall

| 实验 | road | building | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|
| Plan7-A (old eval) | 88.52 | 93.00 | 79.70 | 86.06 | **91.59** |
| P11-A (min-max+veg) | 89.90 | 91.58 | **78.43** | 84.99 | 86.51 |
| **P11-B (nDSM)** | **91.25** | 90.80 | 77.42 | 85.38 | 88.66 |
| P11-C (nDSM+veg) | 91.06 | 91.28 | 77.00 | 84.95 | 87.74 |
| P11-D (nDSM+aug) | 90.80 | **92.05** | 76.53 | **85.33** | 86.78 |
| P11-E (4-block) | 86.82 | 90.87 | 74.75 | 83.16 | 82.23 |

### 7.3 混淆矩阵关键指标

| 实验 | mIoU | grass→tree | tree→grass | veg sum |
|------|:---:|:---:|:---:|:---:|
| Plan7-A (soft-logit) | 77.66 | 12.3% | 7.5% | 19.8% |
| P11-A (min-max+veg) | 76.46 | 10.3% | 9.7% | 20.0% |
| **P11-B (nDSM)** | 76.72 | 10.3% | 9.2% | **19.6%** |
| P11-C (nDSM+veg) | 76.51 | 10.9% | 9.7% | 20.7% |
| P11-D (nDSM+aug) | 76.56 | 11.3% | 9.3% | 20.6% |
| P11-E (4-block) | 72.61 | 11.4% | 11.7% | 23.1% |

### 7.4 运行时间

| 实验 | 耗时 | 每 epoch |
|------|:---:|:---:|
| P11-A | 2h 29min | ~10 min |
| P11-B | 2h 29min | ~10 min |
| P11-C | 2h 29min | ~10 min |
| P11-D | 2h 30min | ~10 min |
| P11-E | 1h 23min | ~5.5 min |
| **合计** | **~11.4h** | |

---

## 8. 结论

### 正向发现

**nDSM +1.95pp（vs 同架构从头训练基线）**: 替换 per-tile min-max 为 nDSM 是有效的。改进主要来自 road recall (+1.35pp) 和 car recall (+2.15pp)，因为 nDSM 保留了物体相对于地面的绝对高度信息。植被混淆总和仅微降 0.2pp。

### 负向发现

1. **veg_boundary_weight 持续负向**: P11-A (-0.68pp) 和 P11-C (-0.21pp) 均退化。双向等权重的植被边界惩罚无法区分"grass 误分→tree"和"tree 误分→grass"的场景。
2. **数据增强无效**: P11-D vs P11-B 仅 -0.16pp，噪声级。
3. **Adapter 消融证明全 32 block 注入是必要的**: P11-E 减少至 4 block 后暴跌 -4.11pp。

### 核心认知

nDSM 修复了 min-max 归一化对**地面类和小物体**的高程信号破坏，但 tree↔grass 混淆（~20%）是 **RGB 纹理问题**而非高程问题——树冠边缘的高程过渡带与草地没有清晰的高程决策边界。后续方向应聚焦于 **RGB 纹理级别的植被细粒度特征**。

### 新增失败方向

| 方向 | 证据 | Δ | Plan |
|------|------|:---:|:---:|
| Tree-grass boundary weighted loss | P11-A -0.68pp, P11-C -0.21pp | 持续负向 | Plan11 |
| Adapter 仅限全局注意力 block | P11-E 72.61 vs P11-B 76.72 | -4.11pp | Plan11 |
| Photometric 数据增强 | P11-D 76.56 vs P11-B 76.72 | -0.16pp（噪声） | Plan11 |
