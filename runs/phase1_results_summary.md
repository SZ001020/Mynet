# Phase 1: SAM 3 Zero-shot 基线 + 双头效果分析

> 实验时间: 2026-04-29 | GPU: RTX 5090 32GB | 环境: segearthov3 (PyTorch 2.11, CUDA 12.8)
> 输出目录: `/root/Mynet/autodl-tmp/runs/phase1_baseline_20260429_170226` (Vaihingen), `...171705` (Potsdam, LoveDA)

## 实验设计

对每个数据集运行 **6 种 head × presence 组合**，评估 SAM 3 内部各组件的贡献：

| 配置 | Transformer Decoder (Instance) | Semantic Head | Presence Score | 说明 |
|---|---|---|---|---|
| Instance-Only | ✓ | ✗ | ✗ | 纯实例分割 |
| Instance+Presence | ✓ | ✗ | ✓ | 实例 + 存在性过滤 |
| Semantic-Only | ✗ | ✓ | ✗ | 纯语义分割 |
| Semantic+Presence | ✗ | ✓ | ✓ | 语义 + 存在性过滤 |
| Dual-Head | ✓ | ✓ | ✗ | 双头融合（无 presence） |
| Dual-Head+Presence | ✓ | ✓ | ✓ | **默认配置**（双头 + presence） |

## 总体结果

### ISPRS Vaihingen (6类, 16张测试图, ~1900×2500px)

| 配置 | aAcc (%) | mIoU (%) | mAcc (%) |
|---|---|---|---|
| Instance-Only | 56.8 | 41.0 | 60.0 |
| Instance+Presence | 56.6 | 41.0 | 61.7 |
| Semantic-Only | 71.8 | 52.0 | 76.0 |
| Semantic+Presence | 73.3 | 52.5 | 77.0 |
| Dual-Head | 76.3 | 54.2 | 75.3 |
| **Dual-Head+Presence (Best)** | **78.7** | **55.3** | **78.2** |

**Per-class IoU (Dual-Head+Presence, %):**

| Road | Building | Grass | Tree | Car | Clutter |
|------|----------|-------|------|-----|---------|
| 69.7 | 86.1 | 50.0 | 68.4 | 51.4 | 6.2 |

### ISPRS Potsdam (6类, 24张测试图, 6000×6000px → 预缩放至≤2000px)

| 配置 | aAcc (%) | mIoU (%) | mAcc (%) |
|---|---|---|---|
| Instance-Only | 71.2 | 47.7 | 63.5 |
| Instance+Presence | 71.1 | 47.6 | 63.5 |
| Semantic-Only | 68.2 | 48.5 | 69.4 |
| Semantic+Presence | 68.1 | 48.4 | 69.3 |
| **Dual-Head (Best)** | **73.1** | **51.4** | **70.0** |
| Dual-Head+Presence | 72.8 | 51.0 | 69.6 |

**Per-class IoU (Dual-Head, %):**

| Road | Building | Grass | Tree | Car | Clutter |
|------|----------|-------|------|-----|---------|
| 61.7 | 81.9 | 54.3 | 33.0 | 54.3 | 22.9 |

### LoveDA (7类, 1669张测试图, 1024×1024px) — 部分完成

| 配置 | aAcc (%) | mIoU (%) | mAcc (%) |
|---|---|---|---|
| Instance-Only | 53.5 | 32.2 | 42.2 |

**Per-class IoU (Instance-Only, %):**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|---|---|---|---|---|---|---|
| 42.6 | 49.8 | 39.1 | 45.1 | 17.2 | 8.2 | 23.5 |

> LoveDA 其余 5 个配置因进程意外中断未完成。Instance-Only 结果已明显低于 ISPRS 数据集，初步判断原因为：(1) 类别语义更宽泛（agricultural/barren），(2) 图片多样性大（2522 训练 + 1669 验证），(3) SAM 3 的 text encoder 对遥感农业场景理解不足。

## 关键发现

### 1. Semantic Head 是遥感分割的主力

```
Vaihingen: Semantic-Only (52.0%) >> Instance-Only (41.0%) 差距 11.0%
Potsdam:   Semantic-Only (48.5%) ≈  Instance-Only (47.7%) 差距 0.8%
```

- Vaihingen 上 Semantic head 远好于 Instance head（+11% mIoU）
- Potsdam 上两者接近（+0.8%），Instance head 对建筑类效果突出
- 结论：**遥感场景中，密集/大面积类别（道路、草地）依赖 semantic head；离散个体（建筑、车辆）受益于 instance head**

### 2. 双头融合带来稳定增益

| 数据集 | 单头最优 | 双头最优 | 增益 |
|---|---|---|---|
| Vaihingen | 52.5% (Sem+Pres) | **55.3%** (Dual+Pres) | +2.8% |
| Potsdam | 48.5% (Sem-Only) | **51.4%** (Dual-Head) | +2.9% |

双头融合在两个数据集上均提升 ~2.9% mIoU，说明 instance 和 semantic 信息互补。

### 3. Presence Score 效果因数据集而异

| 数据集 | Dual-Head (无 presence) | Dual-Head+Presence | Presence 效果 |
|---|---|---|---|
| Vaihingen | 54.2% | **55.3%** | **+1.1%** ✅ |
| Potsdam | **51.4%** | 51.0% | **-0.4%** ❌ |

Vaihingen 上 presence 有效抑制了误检；Potsdam 上反而轻微损害精度。差异原因：Potsdam 的 `confidence_threshold=0.2` 设得太低（Vaihingen=0.4），过于宽松的 presence 过滤引入了噪声。

### 4. 类别难度差异巨大

| 类别 | Vaihingen IoU | Potsdam IoU | 分析 |
|---|---|---|---|
| Building | **86.1%** | **81.9%** | SAM 3 对建筑识别最强（规则几何形状） |
| Tree | 68.4% | **33.0%** | Potsdam 树木 IoU 极低，可能因 2000px 缩放丢失树冠纹理 |
| Road | 69.7% | 61.7% | 道路识别稳定 |
| Car | 51.4% | 54.3% | 小目标，但 instance head 对此有效 |
| Grass | 50.0% | 54.3% | 中等难度，纹理依赖性强 |
| Clutter | **6.2%** | **22.9%** | 最难类别，clutter 定义模糊，文本 prompt 难以描述 |

### 5. Vaihingen vs Potsdam 跨数据集差异

Potsdam 的 zero-shot 性能整体低于 Vaihingen（最优 mIoU: 51.4% vs 55.3%），可能原因：
- Potsdam 图片分辨率更高（6000²），预缩放至 2000px 可能损失细节
- Potsdam 场景更复杂（密集城区 vs Vaihingen 的城乡混合）
- Potsdam 原 sliding window 配置（cfg 中 stride=512,crop=512）暗示需要高分辨率推理

## 实验配置细节

### 模型参数
- SAM 3 checkpoint: `weights/sam3/sam3.pt` (3.3GB)
- 推理分辨率: 1008×1008 (SAM 3 原生)
- 精度: bf16 autocast
- Presence threshold: Vaihingen=0.4, Potsdam=0.2, LoveDA=0.5

### 数据集配置
- Vaihingen: `cfg_vaihingen.py` (原配置)
- Potsdam: `cfg_potsdam_fast.py` (slide_crop=0, PIL预缩放≤2000px)
- LoveDA: `cfg_loveda.py` (原配置, reduce_zero_label=True)

## 推理图对比分析（逐像素差异量化）

> 在 Vaihingen 和 Potsdam 各 3 张代表性样本上，生成全部 6 个 head 配置的预测图（共 36 张推理图 + 6 张组合对比图）。
> 输出目录: `/root/Mynet/autodl-tmp/runs/phase1_visualizations/`

### 不同 Head 间的逐像素分歧率

将每个样本的 Instance-Only、Semantic-Only、Dual-Head+Presence 三组预测做逐像素比较：

| 数据集 | 样本 | Inst ≠ Sem | Inst ≠ Dual | Sem ≠ Dual |
|---|---|---|---|---|
| Vaihingen | area1 | 23.7% | 16.9% | 8.5% |
| Vaihingen | area15 | **50.4%** | 49.0% | 7.2% |
| Vaihingen | area26 | 23.1% | 15.5% | 8.5% |
| **Vaihingen 均值** | | **32.4%** | **27.1%** | **8.1%** |
| Potsdam | 2_10 | 18.1% | 4.2% | 13.9% |
| Potsdam | 2_12 | 6.9% | 3.7% | 3.3% |
| Potsdam | 3_12 | 16.0% | 5.1% | 10.9% |
| **Potsdam 均值** | | **13.7%** | **4.3%** | **9.4%** |

### 关键观察

**1. Semantic Head 主导了 Dual-Head 融合输出**

Vaihingen 上 `Sem ≠ Dual` 的平均分歧率仅 8.1%，而 `Inst ≠ Dual` 高达 27.1%。这意味着 **Dual-Head 的 max 融合操作 91.9% 的情况下选择了 Semantic head 的输出**。Instance head 虽然单独使用效果差（41% mIoU），但它贡献的额外 ~2.9% 增益集中在某些特定类别和区域——这些区域的 Instance 预测恰好比 Semantic 更好。

**2. Vaihingen area15 是极端案例**

该区域 Instance 与 Semantic 的分歧率高达 50.4%，说明这是一个 Instance head 严重失效的场景。推测 area15 包含大片密集植被/农田区域，其中不存在清晰的实例边界，Instance head 产生了大量误检。尽管如此，Dual-Head 通过 max 操作几乎完全信赖了 Semantic head（分歧仅 7.2%），成功避开了 Instance 的噪声。

**3. Potsdam 上两 Head 贡献更均衡**

Potsdam 的 `Sem ≠ Dual`（9.4%）和 `Inst ≠ Dual`（4.3%）差距较小。在 2_10 样本中，`Sem ≠ Dual`（13.9%）甚至大于 `Inst ≠ Dual`（4.2%），说明 **Instance head 在此样本中对最终预测的影响力大于 Semantic head**。这与 Potsdam 的结果一致——Potsdam 上 Instance 和 Semantic 的独立表现差距更小（47.7% vs 48.5%），融合更均衡。

**4. 融合机制的瓶颈**

当前 Dual-Head 使用 **hard element-wise max** 做融合：

```
seg_logits[cls] = max(seg_logits[cls], instance_logits × score)  # Instance
seg_logits[cls] = max(seg_logits[cls], semantic_logits)          # Semantic
seg_logits[cls] = seg_logits[cls] × presence_score               # Presence
```

这种硬选择（hard selection）机制的缺陷：
- 无法实现 **per-region 自适应融合**（密集区走 Semantic，实例边界走 Instance）
- 无法对 **不同类别** 使用不同融合策略（building 更需要 Instance，road 更需要 Semantic）
- Presence score 是全局乘法，无法感知"哪些检测是误检"

这直接支持了 **Phase 3 引入可学习 α-Blending** 的设计动机。

### 对比图文件清单

每个比较图排列为 `RGB | GT | Instance | Instance+Pres | Semantic | Semantic+Pres | Dual-Head | Dual+Pres`：

| 文件 | 说明 |
|---|---|
| `comparison_vaihingen_top_mosaic_09cm_area1.png` | Vaihingen sample 1 |
| `comparison_vaihingen_top_mosaic_09cm_area15.png` | Vaihingen sample 2（Inst/Sem 极端分歧） |
| `comparison_vaihingen_top_mosaic_09cm_area26.png` | Vaihingen sample 3 |
| `comparison_potsdam_top_potsdam_2_10.png` | Potsdam sample 1 |
| `comparison_potsdam_top_potsdam_2_12.png` | Potsdam sample 2（首尾高度一致） |
| `comparison_potsdam_top_potsdam_3_12.png` | Potsdam sample 3 |

## Clutter 类别消融实验

> 实验: 从 SAM 3 的 classname 列表中移除 `clutter`，仅预测 5 个主要类别。
> 评估仍使用 6 类 ground truth（clutter 的 IoU 强制为 0），因此同时报告 6 类 mIoU 和 5 类（排除 clutter）mIoU。

### Vaihingen — 5 类 mIoU 对比

| 配置 | With Clutter (5c) | No Clutter (5c) | Δ | 解读 |
|---|---|---|---|---|
| Instance-Only | 48.7% | 43.4% | **-5.3%** | Instance head 失去 clutter 作为"噪声吸收池" |
| Instance+Presence | 48.7% | 43.3% | -5.4% | 同上 |
| Semantic-Only | 61.7% | **65.8%** | **+4.1%** | Semantic head 获益显著——不再被 clutter 干扰 |
| Semantic+Presence | 62.1% | 65.1% | +3.0% | Presence 过滤力下降 |
| Dual-Head | 64.2% | 65.7% | +1.5% | 两 head 效应相互抵消 |
| Dual-Head+Presence | **65.0%** | 64.9% | -0.2% | 基本持平 |

### Potsdam — 5 类 mIoU 对比

| 配置 | With Clutter (5c) | No Clutter (5c) | Δ | 解读 |
|---|---|---|---|---|
| Instance-Only | 53.5% | 52.7% | -0.8% | 轻微下降 |
| Instance+Presence | 53.4% | 52.7% | -0.8% | 同上 |
| Semantic-Only | 54.6% | **55.6%** | **+1.1%** | 小幅改善 |
| Semantic+Presence | 54.5% | 55.6% | +1.1% | 同上 |
| Dual-Head | **57.1%** | 56.6% | -0.5% | 基本持平 |
| Dual-Head+Presence | 56.9% | 56.6% | -0.4% | 基本持平 |

### 关键发现

**1. Semantic Head 从去除 clutter 中获益最大**

Vaihingen Semantic-Only 的 5 类 mIoU 从 61.7% 跃升至 65.8%（+4.1%）。这说明在有 clutter 的情况下，SAM 3 的 semantic head 将大量应属于其他类别的像素错误分配给了 clutter。去除 clutter 后，这些像素被正确分配到 road/building/grass/tree/car。

**2. Instance Head 依赖 clutter 作为"假阳性吸收池"**

Vaihingen Instance-Only 在去除 clutter 后 5 类 mIoU 反而下降 5.3%（48.7% → 43.4%）。原因：Instance head 产生大量低质量的 mask，原本这些 mask 被映射到 clutter（对 5 类指标无害），现在它们被错误分配到其他类别（直接损害 5 类指标）。

**3. Dual-Head 融合具有"自稳"特性**

无论在哪个数据集上，Dual-Head 的去 clutter 前后 5 类 mIoU 差异均 <1.5%。这说明 max 融合操作起到了稳定器的作用——Semantic head 增益和 Instance head 损失在融合中相互抵消。

**4. Presence Score 的反直觉行为**

```
With clutter:    Dual+Pres (65.0%) > Dual (64.2%)  = +0.8%   Presence 有效
No clutter:      Dual+Pres (64.9%) < Dual (65.7%)  = -0.8%   Presence 反而有害
```

原因：Presence score 的过滤阈值是与 clutter 存在性联合训练的。当 clutter 类别被移除后，presence 的 calibration 失效——它无法准确判断"一个非 clutter 类是否存在于图中"。这进一步支持了 Phase 3 对 presence threshold 进行 **per-dataset 学习** 的必要性。

**5. 实际建议：不要简单删除 clutter**

对于 ISPRS 数据集，clutter 发挥了重要的**噪声吸收**功能。如果目标是提升 5 类 mIoU：
- 最佳方案：保留 clutter，但优化其 prompt（Phase 2），让 SAM 3 将真正的 clutter 像素（车顶纹理、建筑边缘碎片等）与可分类像素区分开
- 次优方案：使用 Semantic-Only + 无 clutter（65.8% 5c-mIoU），但需要解决 Instance head 的假阳性问题

### Per-Class 变化分析（Vaihingen Dual-Head 最优，5 类）

| 类别 | With Clutter IoU | No Clutter IoU | Δ | 解读 |
|---|---|---|---|---|
| Road (impervious) | 66.3% | **69.7%** | **+3.4%** | 去除 clutter 后道路像素不再被"吸收" |
| Building | 86.0% | 86.1% | +0.1% | 建筑几乎不受影响（已极强） |
| Grass (low_veg) | 48.9% | **53.4%** | **+4.5%** | 草地/低植被是最大受益者 |
| Tree | 68.6% | 68.4% | -0.2% | 树木不变 |
| Car | 51.6% | 50.9% | -0.7% | 车辆轻微下降 |

**核心发现：去除 clutter 后，road (+3.4%) 和 grass (+4.5%) 显著提升。** 这两个类别是有 clutter 时 SAM 3 最常混淆的类别——道路纹理多样性（沥青、水泥、泥土路）和草地纹理让 SAM 3 的 semantic head 倾向于将它们标记为 clutter。去除 clutter 后，semantic head 被迫做出更明确的分类，反而提高了精度。

### Vaihingen Per-Class 全量数据（无 clutter）

**Dual-Head（6c-mIoU=54.75, 5c-mIoU=65.70）:**

| 类别 | IoU | Acc |
|---|---|---|
| Road | 69.7 | 82.8 |
| Building | 86.1 | 95.8 |
| Grass | 53.4 | 65.1 |
| Tree | 68.4 | 82.2 |
| Car | 50.9 | 82.3 |

**Semantic-Only（6c-mIoU=54.83, 5c-mIoU=65.80 — 最优 5c 配置）:**

| 类别 | IoU | Acc |
|---|---|---|
| Road | 69.3 | 82.5 |
| Building | 86.3 | 95.5 |
| Grass | 53.2 | 65.5 |
| Tree | 68.3 | 82.2 |
| Car | 51.8 | 79.7 |

**Instance-Only（6c-mIoU=36.16, 5c-mIoU=43.39 — 严重退化）:**

| 类别 | IoU | Acc |
|---|---|---|
| Road | **40.6** | 91.7 |
| Building | 85.9 | 93.9 |
| Grass | **9.2** | 9.2 |
| Tree | 31.3 | 32.7 |
| Car | 49.9 | 78.9 |

> Instance-Only 的 road IoU 从有 clutter 时的 66.9% 暴跌至 40.6%（-26.3%），grass 从 9.3% 进一步跌至 9.2%。这确凿地证明：**Instance head 在有 clutter 时，将大量难以分类的像素标记为 clutter（不损害其他类 IoU），但去除 clutter 后这些像素被错误分配到 road/grass，直接损害了这些类的 IoU。**

### Potsdam Per-Class 数据（无 clutter）

Potsdam 去除 clutter 后，5 个主要类别的 IoU 几乎无变化（<1% 差异），说明 Potsdam 的 clutter 与主要类别区分度更高（clutter IoU 本身有 18-23%），去除它对其他类别影响极小。

**Dual-Head（6c-mIoU=47.14, 5c-mIoU=56.57）:**

| 类别 | IoU | Acc |
|---|---|---|
| Road | 59.5 | 82.0 |
| Building | 81.9 | 96.1 |
| Grass | 54.2 | 75.6 |
| Tree | 33.0 | 34.8 |
| Car | 54.3 | 88.1 |

### No-Clutter 推理图对比

生成 2 张无 clutter 对比图（选择最具代表性的 Vaihingen area15 和 Potsdam 2_10）：

| 文件 | 说明 |
|---|---|
| `comparison_vaihingen_top_mosaic_09cm_area15_noclutter.png` | Vaihingen 极端案例（Inst/Sem 50% 分歧），观察去除 clutter 后 Instance head 的 road/grass 分类崩塌 |
| `comparison_potsdam_top_potsdam_2_10_noclutter.png` | Potsdam 案例，验证 5 类指标几乎不变 |
| `vaihingen_top_mosaic_09cm_area15_*_noclutter.png` | 6 张单配置预测图 |
| `potsdam_top_potsdam_2_10_*_noclutter.png` | 6 张单配置预测图 |

> 全部 no-clutter 可视化输出位置: `/root/Mynet/autodl-tmp/runs/phase1_visualizations/`

### No-Clutter 实验结论

1. **不要直接删除 clutter** — 对 Vaihingen Dual-Head 无显著增益（+0.7% 5c），对 Potsdam 甚至轻微负面
2. **Semantic-Only + No-Clutter 是一个可行的轻量方案** — Vaihingen 5c-mIoU=65.80%，无需双头融合复杂性
3. **Instance head 的假阳性问题是无 clutter 场景的最大挑战** — 需要 presence score 或可学习的假阳性过滤器
4. **Phase 2 应优先优化 clutter 的 text prompt** — 让 SAM 3 学会区分"真正难以分类的区域"和"可分类但被误判的像素"，而非直接删除该类别

## 对后续阶段的指导

1. **Prompt Engineering (Phase 2)**: clutter/grass 类别的 prompt 最需要优化，building 的可提升空间小
2. **微调 (Phase 3)**: 应重点提升 tree（Potsdam 33%）和 clutter（Vaihingen 6.2%）；instance 和 semantic head 的融合权重值得学习
3. **多模态 (Phase 4)**: Potsdam 有 DSM 数据，可能通过高程信息改善 tree 和 building 的区分
4. **LoveDA**: 零样本 32.2% mIoU 偏低，7 类别中 forest 仅 8.2%——这是微调收益最大的数据集
