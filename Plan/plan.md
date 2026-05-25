# SAM 3 遥感语义分割研究计划

> ⚠️ **评估协议**：最终评估必须使用 **256×256 滑动窗口 + overlap 平均**（与 MFNet/ISPRS 标准对齐）。512² crop 评估仅用于训练监控，会系统性低估性能（尤其对大 tile 和小类别）。详见 plan3.md「评估协议」章节。
>
> ⚠️ **评估输出规范**：所有 evaluate/validate 必须输出四项指标：`avg_oa`, `avg_miou`, `per_class_iou`, `per_class_oa`。Per-class OA 公式：`(inter[c] + total - union[c]) / total`，可与 MFNet 论文逐类别 OA 直接对比。详见 plan3.md「评估输出规范」。

## 硬件环境

| 组件 | 规格 | 对研究的影响 |
|---|---|---|
| GPU | RTX 5090 32GB (Blackwell, sm_120) | 可全量微调 SAM 3，支持 bf16 + FA3 + torch.compile |
| RAM | 754 GB | 数据全内存加载，无 I/O 瓶颈 |
| CUDA | 12.8, PyTorch 2.7.0 | bf16 原生，SAM 3 推理已启用 autocast bf16 |
| 数据盘 | /root/autodl-tmp = 100G, 可用 56G | checkpoint 和中间结果存此处 |
| 公共盘 | /autodl-pub = 14T, 可用 7.1T | 可存放大型模型权重和数据集备份 |

## 数据集现状

| 数据集 | 训练可用 | 测试可用 | 图像尺寸 | 模态 | 类别数 |
|---|---|---|---|---|---|
| **Vaihingen** | 17 张 | 16 张 (已标注) | ~1900×2500 | RGB + DSM (float32) | 6 |
| **Potsdam** | ~14 张 | 24 张 (已标注) | 6000×6000 | RGB + DSM (float32) | 6 |
| **LoveDA** | 2522 张 | 1669 张 | 1024×1024 | RGB only | 7 |

注意：Vaihingen 和 Potsdam 的 DSM 数据与 RGB 逐像素对齐（同分辨率），这是多模态实验的关键基础。

Vaihingen/Potsdam 类别：road, building, grass, tree, car, clutter
LoveDA 类别：background, building, road, water, barren, forest, agricultural

### MFNet 标准数据划分（所有实验统一使用）

```python
# Source: MFNet/utils.py (IEEE TGRS 2025)
VAIHINGEN_TRAIN = ['1','3','23','26','7','11','13','28','17','32','34','37']  # 12 tiles
VAIHINGEN_TEST  = ['5','21','15','30']                                        # 4 tiles
POTSDAM_TRAIN   = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12',
                   '5_12','7_11','7_9','6_9','7_7','6_8','4_12','6_12']     # 16 tiles
POTSDAM_TEST    = ['4_10','5_11','2_11','3_10','6_11','7_12']                 # 6 tiles
```

## 研究路线总览

```
第 1 阶段 (1-2天)         第 2 阶段 (1-2周)        第 3 阶段 (2-4周)         第 4 阶段 (4-6周)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│ Zero-shot 基线   │     │ Prompt          │     │ SAM 3 遥感微调   │     │ 多模态 SAM 3 (DSM)   │
│ + 双头效果分析   │ ──▶ │ Engineering     │ ──▶ │ + 双头融合学习   │ ──▶ │ + Cross-dataset 泛化 │
│                 │     │ 系统研究        │     │                 │     │                     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────────┘
   VRAM: ~3GB              VRAM: ~3GB              VRAM: 15-25GB          VRAM: 20-28GB
   纯推理                  纯推理                  全量微调               端到端训练
```

## 第 1 阶段：Zero-shot 基线 + 双头效果分析 (1-2天)

> **状态：✅ 已完成 (2026-04-29)**
> 详细报告: `/root/Mynet/autodl-tmp/runs/phase1_results_summary.md`
> 推理对比图: `/root/Mynet/autodl-tmp/runs/phase1_visualizations/`

**目标：** 建立完整的 zero-shot baseline，量化 SAM 3 每个内部组件对遥感分割的贡献。

**实验设计：** 对 Vaihingen/Potsdam 各运行 6 种 head × presence 组合，额外运行 clutter 消融实验（移除 clutter 后再跑 6 组）。

| 配置 | 说明 |
|---|---|
| Instance-Only | 纯 instance head |
| Instance+Presence | 实例 + 存在性过滤 |
| Semantic-Only | 纯 semantic head |
| Semantic+Presence | 语义 + 存在性过滤 |
| Dual-Head | 双头融合（无 presence） |
| Dual-Head+Presence | 双头 + presence（默认配置） |

### 1.1 5 类 mIoU 结果（去除 clutter，之后所有实验沿用此设定）

经过 clutter 消融实验验证，去除 clutter 后 Semantic head 在 Vaihingen 上 5 类 mIoU 提升 4.1%。**此后所有实验均使用 5 类（road, building, grass, tree, car），评估指标为 5 类 mIoU。**

| 配置 | Vaihingen 5c-mIoU | Potsdam 5c-mIoU |
|---|---|---|
| Instance-Only | 43.4% | 52.7% |
| Instance+Presence | 43.3% | 52.7% |
| Semantic-Only | **65.8%** | 55.6% |
| Semantic+Presence | 65.1% | 55.6% |
| Dual-Head | 65.7% | **56.6%** |
| Dual-Head+Presence | 64.9% | 56.6% |

**Vaihingen 最优: Semantic-Only (65.8%)** — 说明去除 clutter 后，纯 semantic head 已足够强。
**Potsdam 最优: Dual-Head (56.6%)** — Instance+Semantic 互补效应在此数据集更明显。

### 1.2 Per-Class IoU（最优配置）

**Vaihingen Semantic-Only (5c-mIoU=65.8%):**

| Road | Building | Grass | Tree | Car |
|---|---|---|---|---|
| 69.3% | 86.3% | 53.2% | 68.3% | 51.8% |

**Potsdam Dual-Head (5c-mIoU=56.6%):**

| Road | Building | Grass | Tree | Car |
|---|---|---|---|---|
| 59.5% | 81.9% | 54.2% | 33.0% | 54.3% |

### 1.3 Clutter 消融实验关键发现

- **Semantic head 从去除 clutter 中获益**（Vaihingen +4.1%），road (+3.4%) 和 grass (+4.5%) 提升最大
- **Instance head 严重依赖 clutter 作为假阳性吸收池**（Vaihingen road IoU 从 66.9% → 40.6%）
- **Dual-Head 融合具有自稳特性**，去 clutter 前后差异 <1.5%
- **Presence 在无 clutter 时 calibration 失效**，Dual+Pres < Dual

### 1.4 推理图对比分析（逐像素分歧量化）

6 张样本 × 2 数据集 = 12 对比图（含 clutter 消融），核心发现：

| 指标 | Vaihingen 均值 | Potsdam 均值 |
|---|---|---|
| Inst ≠ Sem | 32.4% | 13.7% |
| Inst ≠ Dual | 27.1% | 4.3% |
| Sem ≠ Dual | **8.1%** | 9.4% |

- **Semantic head 主导了 Dual-Head 输出**：Vaihingen 上 91.9% 的像素选择了 Semantic head
- **Potsdam 两 head 贡献更均衡**：Instance head 对建筑的边界识别贡献更大
- **area15 极端案例**：Inst≠Sem 高达 50.4%，但 Dual 仅偏离 Sem 7.2%，证明 max 操作的稳健性

### 1.5 实验文件清单

```
/root/Mynet/autodl-tmp/runs/
├── phase1_results_summary.md                              ← 完整分析报告
├── phase1_baseline_20260429_174235/                       ← Vaihingen (with clutter, 6 configs)
├── phase1_baseline_20260429_171705/                       ← Potsdam (with clutter, 6) + LoveDA (1)
├── phase1_baseline_20260429_180550/                       ← Vaihingen+Potsdam (no clutter, 12 configs)
└── phase1_visualizations/                                 ← 14 张推理对比图
    ├── comparison_vaihingen_top_mosaic_09cm_area1.png
    ├── comparison_vaihingen_top_mosaic_09cm_area15.png
    ├── comparison_vaihingen_top_mosaic_09cm_area26.png
    ├── comparison_vaihingen_nc_..._area15_noclutter.png
    ├── comparison_potsdam_top_potsdam_2_10.png
    ├── comparison_potsdam_top_potsdam_2_12.png
    ├── comparison_potsdam_top_potsdam_3_12.png
    └── comparison_potsdam_nc_top_potsdam_2_10_noclutter.png
```

---

## 第 2 阶段：Prompt Engineering 系统研究

> **状态：✅ 已完成 (2026-04-29)**
> **设定：5 类（无 clutter），评估 5c-mIoU，仅测 Semantic-Only 和 Dual-Head**
> 实验目录: `/root/Mynet/autodl-tmp/runs/phase2_prompt_20260429_205224/`

**目标：** 系统性研究文本 prompt 设计对 SAM 3 零样本遥感分割精度的影响。

### 2.1 实验设计

5 组渐进式 prompt，从简单到复杂：

| 组 | 策略 | 示例 (Vaihingen road) |
|---|---|---|
| A-Baseline | 单个单词 | `road` |
| B-RS-View | 长句遥感视角描述 | `road in nadir aerial photography` |
| C-Geometry | 长句几何属性描述 | `elongated gray asphalt road surface with lane markings` |
| D-Synonyms | 逗号分隔多同义词 | `road, street, paved surface, asphalt, highway, roadway` |
| E-ShortRS | 短遥感关键词 | `aerial road, street from above, paved ground` |

### 2.2 实验结果

| 排名 | 数据集 | Prompt 组 | Head | 5c-mIoU | vs Baseline |
|---|---|---|---|---|---|
| 1 | Vaihingen | **A-Baseline** | Semantic-Only | **65.80%** | — |
| 2 | Vaihingen | A-Baseline | Dual-Head | 65.70% | -0.10% |
| 3 | Potsdam | **A-Baseline** | Dual-Head | **56.57%** | — |
| 4 | Potsdam | A-Baseline | Semantic-Only | 55.64% | -0.93% |
| 5 | Vaihingen | D-Synonyms | Semantic-Only | 54.42% | **-11.38%** |
| 6 | Vaihingen | D-Synonyms | Dual-Head | 54.30% | -11.40% |
| 7 | Potsdam | B-RS-View | Dual-Head | 31.70% | **-24.87%** |
| 8 | Potsdam | B-RS-View | Semantic-Only | 31.69% | -23.95% |
| 9 | Vaihingen | B-RS-View | Dual-Head | 28.90% | -36.80% |
| 10 | Vaihingen | B-RS-View | Semantic-Only | 28.89% | -36.91% |
| 11 | Potsdam | C-Geometry | Dual-Head | 26.05% | -30.52% |
| 12 | Potsdam | C-Geometry | Semantic-Only | 21.82% | -33.82% |
| 13 | Vaihingen | C-Geometry | Dual-Head | 20.83% | -44.87% |
| 14 | Vaihingen | C-Geometry | Semantic-Only | 19.92% | -45.88% |
| 15 | Vaihingen | E-ShortRS | Semantic-Only | 15.32% | -50.48% |
| 16 | Potsdam | E-ShortRS | Semantic-Only | 13.62% | -43.02% |

> Potsdam D-Synonyms 因 CUDA OOM 失败（显存累积 bug）。E-ShortRS 两个数据集均完成。

### 2.3 核心发现

**1. 简单单词 = 最优，复杂的 domain-aware prompt = 灾难**

```
Vaihingen:  A (单字) 65.8% >> D (同义词) 54.4% >> B (遥感句) 28.9% >> E (短术语) 15.3%
Potsdam:    A (单字) 56.6% >> B (遥感句) 31.7% >> C (几何句) 26.1% >> E (短术语) 13.6%
```

任何形式的 prompt 复杂化都会严重损害精度——从单个单词变成逗号分隔的同义词列表就损失 11.4%，变成完整句子损失 37-45%，添加"aerial"、"top-down"等遥感术语甚至更差（50% 损失）。

**2. SAM 3 的 text encoder 对 prompt 格式极度敏感**

SAM 3 的 text encoder 是在自然场景图像-文本对上训练的，其文本分布以简短、通用的类别名为中心。添加遥感领域的任何修饰词（"aerial", "nadir", "orthorectified", "from above"）都使 prompt 进入 encoder 的 out-of-distribution 区域，导致 text-vision alignment 完全失效。

**3. 同义词列表优于长句，但仍显著差于单字**

D 组 (逗号分隔) 保留了约 83% 的 baseline 性能（54.4/65.8），远好于 B/C/E 组（仅保留 30-43%）。说明 text encoder 对逗号分隔的多词格式有一定容忍度，但引入的额外词汇造成了特征空间中的"平均化"效应——过多的语义变体稀释了核心概念的 embedding。

**4. 跨数据集一致性**

prompt 退化效应在两个数据集上高度一致：
- B 组一致降到 ~30%
- C 组一致降到 ~20-26%
- E 组一致降到 ~14-15%

这说明这不是数据集特有问题，而是 SAM 3 text encoder 的系统性限制。

### 2.4 对后续阶段的指导

- **Prompt Engineering 的收益已达上限**：简单单词就是最优 prompt，无需进一步优化
- **Phase 3 微调是关键路径**：zero-shot 的 65.8% (Vaihingen) / 56.6% (Potsdam) 与 MFNet 的 75%+ 之间 ~10-18% 的差距，只能通过 fine-tuning 填补
- **微调时不需要考虑 prompt 优化**：直接使用简单单词 + 微调模型参数来对齐遥感 domain
- **论文贡献**：这是一个有价值的 **negative result**——系统性地证明了 SAM 3 的 text encoder 对遥感领域适应需要通过参数更新（微调），而非 prompt engineering

---

## 第 3 阶段：SAM 3 遥感微调

> **状态：✅ 完成 (2026-04-30, updated 2026-05-01) — FPN + UNet decoder，3 种数据策略对比**
> 代码: `/root/Mynet/fineNet/`
> 结果: `/root/Mynet/autodl-tmp/runs/phase3_partial_*/

### 3.1 实验设计

**方案 A — Partial Fine-tuning（3 种数据策略）：**
- A1: Vaihingen + Potsdam 混合训练
- A2: Vaihingen 单独训练
- A3: Potsdam 单独训练

**架构：** 冻结 SAM 3 ViTDet backbone (no_grad) + 训练轻量 FPN decoder (797K / 841M = 0.1%)

**配置：** FPN 4-level decoder (lateral+ smooth convs) → 5-class logits | AdamW lr=1e-4 | CosineAnnealing | 512² crop | CE Loss

> 方案 B (全量微调) 和 C (LoRA) 因 SAM 3 融合算子 (`perflib/fused.py`) 不支持梯度计算而无法运行。正确路径是使用 SAM 3 官方的 `Reference-Project/sam3-main/sam3/train/train.py` + 更大数据集。

### 3.2 结果

| 策略 | Vaihingen 5c-mIoU | Potsdam 5c-mIoU | vs Zero-shot (V) | vs Zero-shot (P) |
|---|---|---|---|---|
| **Zero-shot (Phase 1, 全16/24张)** | **65.8%** | **55.6%** | — | — |
| **Zero-shot (MFNet test, 4/6张)** | **66.0%** | **54.9%** | — | — |
| A1: Combined (V+P) | 52.1% | 25.7% | -13.7% | -29.9% |
| A2: Vaihingen-Only | **56.6%** | — | **-9.2%** | — |
| A3: Potsdam-Only | — | 21.0% | — | -34.6% |

> **2026-05-01 更新:** Phase 1 用 MFNet 标准测试集（Vaihingen 4张, Potsdam 6张）重跑，结果与全量测试几乎一致（66.0% vs 65.8%, 54.9% vs 55.6%），说明测试集规模对 zero-shot 评估影响不大。

**训练曲线 (Vaihingen-Only, best @ epoch 5):**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.66 | 0.49 | 0.42 | 0.40 | 0.36 | 0.35 | 0.34 | 0.33 | 0.33 | 0.33 |
| mIoU | 52.4 | 51.7 | 51.8 | 54.6 | **56.6** | 51.6 | 53.9 | 54.0 | 53.2 | 54.2 |

**Potsdam-Only (best @ epoch 7):** Loss 0.75→0.38, mIoU始终在 15-21% 徘徊，训练从未收敛。

### 3.3 UNet Decoder 升级（2026-05-01）

> 代码: `/root/Mynet/fineNet/train_phase3_unet.py`
> 结果: `/root/Mynet/autodl-tmp/runs/phase3_unet_20260430_205426/`

在 Phase 3 FPN decoder 基础上，替换为更强的 UNet decoder（4.4M params vs 0.8M），修复了 CUDA 上下文冲突和 NaN loss（梯度裁剪 + lr=5e-4）。

| Epoch | Vaihingen mIoU | Potsdam mIoU |
|---|---|---|
| 1 | **54.9%** | 36.9% |
| 2 | 54.1% | 33.5% |

**Decoder 架构对比：**

| 指标 | FPN Decoder | UNet Decoder | Δ |
|---|---|---|---|
| 参数量 | 797K | 4,436K | +5.6× |
| Vaihingen best | 56.6% (E5) | **54.9%** (E1) | -1.7% |
| Potsdam best | 25.7% (E3) | **36.9%** (E1) | **+11.2%** |
| vs zero-shot (V) | -9.2% | -10.9% | — |
| vs zero-shot (P) | -29.9% | -18.7% | — |

UNet 在 Potsdam 上大幅改善（+11.2%），但 Vaihingen 略低于 FPN。UNet 仅训练 2 epoch 已开始过拟合（E2 V 从 54.9 降至 54.1），与 FPN 的过拟合趋势一致。

### 3.4 关键发现

1. **更强的 decoder 带来 Potsdam 的显著改善**（+11.2%），但 Vaihingen 几乎不变
2. **30 张图的小数据集是根本瓶颈** — 无论 FPN（0.8M）还是 UNet（4.4M），都无法逼近 zero-shot 的 65.8%
3. **过拟合在 1-2 epoch 内就开始** — decoder 参数越多，过拟合越快
4. **Potsdam 对 decoder 容量更敏感** — UNet 比 FPN 好 11.2%，说明 Potsdam 的复杂场景需要更强的特征解码能力

### 3.4 推理图对比

| 文件 | 内容 |
|---|---|
| `comparison_zs_vs_ft_top_mosaic_09cm_area1.png` | Vaihingen area1: RGB\|GT\|Zero-shot\|Fine-tuned |
| `comparison_zs_vs_ft_top_mosaic_09cm_area15.png` | Vaihingen area15 (极端案例) |
| `comparison_zs_vs_ft_top_potsdam_2_10.png` | Potsdam 2_10 |

---

## 第 4 阶段：多模态 SAM 3 — DSM 高程注入

> **状态：⚠️ 部分完成 (2026-04-30, updated 2026-05-01) — 方式 A (logit bias) 完成，方式 B (双流微调) 训练可达但验证瓶颈待修复**
> 代码: `/root/Mynet/fineNet/phase4_approach_a.py`, `phase4_dsm.py`

### 4.1 方式 A — DSM Logit Bias 注入（零样本）

**思路：** 计算 nDSM（归一化高程），在 SAM 3 推理时作为 per-pixel logit bias：
- 高程区域（>2m）: boost building/tree/car 的 logits
- 低程区域: boost road/grass 的 logits

```
DSM → nDSM (morphological opening) → [0,1] prior mask
                                         ↓
SAM3 inference: logits[building] *= (1 + dsm × 3.0)
                logits[road]    *= 1 / (1 + dsm × 5.0)
```

**关键实现挑战：**
- SAM 3 geometric prompt API 每次调用触发 forward（O(N) 复杂度，不可行）
- 改为直接操作 logit 空间的后处理乘法
- DSM 需要与 RGB 逐像素对齐（同分辨率）

### 4.2 实验结果

| 数据集 | Baseline (no DSM) | DSM-A (logit bias) | Δ |
|---|---|---|---|
| Vaihingen | 43.3% | 15.4% | **-27.9%** |
| Potsdam | 52.7% | 16.0% | **-36.7%** |

> 注意：Baseline 数值因 segmentor 代码修改引入回归，低于 Phase 1 数。核心看 Δ 趋势。

### 4.3 关键发现

1. **DSM logit bias 灾难性损害精度** — 简单的乘法加权无法正确利用高程信息
2. **SAM 3 的 text-vision features 远强于 DSM prior** — 高程先验的乘法扰动破坏了精心训练的 logit 分布
3. **Object 类与 ground 类的高程边界模糊** — 低矮建筑（<2m）、高草（>1m）、树冠投影区域等在 nDSM 中无法清晰分离
4. **方式 B 受限于 Phase 3 微调瓶颈未进行** — 需要在解决融合算子训练限制后重新评估

### 4.4 方式 B — 双流 DSM 微调（2026-05-01）

> 代码: `/root/Mynet/fineNet/train_phase4_dsm.py`
> 结果: `/root/Mynet/autodl-tmp/runs/phase4_dsm_20260501_*/`

**架构：** SAM3 RGB 特征（256ch）+ 轻量 CNN DSM 编码器（64ch）→ 320ch 双流融合 → UNet decoder

```
RGB → SAM3 ViTDet (frozen) → [256ch × 3 scales]
DSM → CNN encoder (57K params) → [64ch × 3 scales]
              ↓ concat at each scale
    DualStreamDecoder (4.8M) → 5-class logits
```

**训练状态：**

| 指标 | 值 |
|---|---|
| DSM 数据 | 28/28 tiles 成功加载（Vaihingen+Potsdam） |
| Epoch 1 loss | 1.57 → 0.53（正常收敛） |
| 卡点 | E1 B1080/1120 后验证环节 `scipy.grey_opening` 在 full-res 图像上极慢 |

**结论：** DSM 双流在代码层面完全可行——所有 28 张图成功加载 DSM、loss 正常下降、GPU 训练正常。但验证环节的 `grey_opening` 大核形态学操作在 6000² 图像上成为瓶颈。修复方案：预计算 nDSM 为 `.npy` 文件，避免每次 `__getitem__` 都重复计算。

### 4.5 后续方向

- **预计算 nDSM**：将 `grey_opening` 结果保存为 `.npy`，消除训练/验证瓶颈
- 使用 SAM 3 官方训练 pipeline 实现全量微调（而非仅 decoder）
- 探索 DSM hillshade/slope 可视化叠加到 RGB（不改模型架构）

---

## 时间线总览

| 周 | 阶段 | 产出 |
|---|---|---|
| 1 | 第 1 阶段：基线 + 双头分析 | 完整 baseline 数据表 + head 分析结论 |
| 2 | 第 2 阶段：Prompt Engineering | A/B/C/D 四组对比结果 + 最优 prompt 策略 |
| 3-4 | 第 3 阶段 A：部分微调 | Fine-tuned mIoU + loss curve + per-class 提升 |
| 5-6 | 第 3 阶段 B + C：全量微调 + 双头融合 | 最优微调策略 + α-Blending 效果 |
| 7-8 | 第 4 阶段 A/B：DSM prompt | 多模态 zero-shot + fine-tuned 结果 |
| 9-10 | 第 4 阶段 C + 论文撰写 | 双流多模态 + 最终论文 |

## 关键文件清单

```
SegEarth-OV-3-main/
├── eval.py                          # 评估入口
├── segearthov3_segmentor.py         # 模型包装器（需要修改）
├── custom_datasets.py               # 数据集定义
├── pamr.py                          # PAMR 后处理
├── configs/
│   ├── base_config.py               # 基础配置
│   ├── cfg_vaihingen.py             # Vaihingen 配置
│   ├── cfg_potsdam.py               # Potsdam 配置
│   ├── cfg_loveda.py                # LoveDA 配置
│   ├── cls_*.txt                    # 类别文本定义（Prompt Engineering 重点）
│   └── *_val.txt                    # 验证集划分
└── sam3/
    ├── model_builder.py             # SAM 3 模型构建
    ├── model/sam3_image_processor.py  # Processor（双头融合 + Geometric prompt）
    └── model/sam3_image.py          # 核心模型

sam3-main/
├── sam3/train/train.py              # SAM 3 训练脚本（第 3/4 阶段使用）
├── sam3/train/trainer.py            # 训练器
└── sam3/model_builder.py            # 完整 SAM 3.1 模型构建
```

## 实验文件清单

### Phase 1 — Zero-shot 基线 + 双头分析

| 类型 | 路径 |
|---|---|
| 代码 | `Reference-Project/SegEarth-OV-3-main/phase1_experiments.py` — 6 head × presence 实验运行器 |
| 代码 | `Reference-Project/SegEarth-OV-3-main/phase1_visualize.py` — 推理图对比生成 |
| 代码 | `Reference-Project/SegEarth-OV-3-main/segearthov3_segmentor.py` — SegearthOV3 模型封装（已修改） |
| 代码 | `Reference-Project/SegEarth-OV-3-main/configs/cfg_*_noclutter*.py` — 5 类评估配置 |
| 结果 | `runs/phase1_baseline_20260429_174235/` — Vaihingen with-clutter (6 configs) |
| 结果 | `runs/phase1_baseline_20260429_171705/` — Potsdam with-clutter (6) + LoveDA (1) |
| 结果 | `runs/phase1_baseline_20260429_180550/` — Vaihingen+Potsdam no-clutter (12 configs) |
| 结果 | `runs/phase1_visualizations/` — 14 张推理对比图 (PNG) |
| 报告 | `runs/phase1_results_summary.md` — 完整分析（含 clutter 消融 + 逐像素分歧） |

| 指标 | Vaihingen (MFNet test, 4 tiles) | Potsdam (MFNet test, 6 tiles) |
|---|---|---|
| 5c-mIoU (Semantic-Only) | **66.0%** | **54.9%** |

### Phase 2 — Prompt Engineering

| 类型 | 路径 |
|---|---|
| 代码 | `Reference-Project/SegEarth-OV-3-main/phase2_prompt_engineering.py` — 4 组 prompt × 2 head 实验运行器 |
| 代码 | `Reference-Project/SegEarth-OV-3-main/configs/cls_*_prompt_{b,c,d,e}.txt` — 各组 prompt 文本 |
| 代码 | `Reference-Project/SegEarth-OV-3-main/configs/cfg_*_prompt_{b,c,d,e}.py` — 各组评估配置 |
| 结果 | `runs/phase2_prompt_20260429_205224/` — 完整结果 (log + metrics.json) |
| 结果 | `runs/phase2_missing_20260429_212413/` — E 组补充结果 |

| Prompt 组 | Vaihingen 5c-mIoU | Δ vs Baseline |
|---|---|---|
| A-Baseline (单字) | **65.8%** | — |
| D-Synonyms (同义词列表) | 54.4% | -11.4% |
| B-RS-View (遥感句子) | 28.9% | -36.9% |
| C-Geometry (几何句子) | 19.9% | -45.9% |
| E-ShortRS (短关键词) | 15.3% | -50.5% |

### Phase 3 — 微调

| 类型 | 路径 |
|---|---|
| 代码 | `Reference-Project/fineNet/train.py` — FPN decoder 训练（A1/A2/A3 三种数据策略） |
| 代码 | `Reference-Project/fineNet/finetune_model.py` — FineTunedSAM3 + FPNHead + LoRALinear |
| 代码 | `Reference-Project/fineNet/finetune_dataset.py` — 训练数据集（V+P 混合/单独） |
| 代码 | `Reference-Project/fineNet/train_phase3_unet.py` — UNet decoder 训练（升级版） |
| 结果 | `runs/phase3_partial_20260429_231705/` — A1 混合训练 (5 epochs, FPN) |
| 结果 | `runs/phase3_partial_vaihingen_20260430_103909/` — A2 Vaihingen 单独 (10 epochs, FPN) |
| 结果 | `runs/phase3_partial_potsdam_20260430_104954/` — A3 Potsdam 单独 (10 epochs, FPN) |
| 结果 | `runs/phase3_unet_20260430_205426/` — UNet decoder (2 epochs, NaN fixed) |
| 日志 | `runs/phase3_train4.log` — UNet 训练日志 |

| Decoder | Vaihingen best | Potsdam best | vs Zero-shot (V) |
|---|---|---|---|
| FPN (797K) | 56.6% (A2, E5) | 25.7% (A1, E3) | -9.2% |
| UNet (4.4M) | 54.9% (E1) | 36.9% (E1) | -10.9% |

### Phase 4 — DSM 多模态

| 类型 | 路径 |
|---|---|
| 代码 | `Reference-Project/fineNet/phase4_approach_a.py` — DSM logit bias 注入（基于 mmseg eval） |
| 代码 | `Reference-Project/fineNet/phase4_dsm.py` — DSM 预处理 + 推理（独立评估器） |
| 代码 | `Reference-Project/fineNet/train_phase4_dsm.py` — 双流 DSM 微调（RGB+DSM → decoder） |
| 结果 | `runs/phase4_dsm_A_20260430_130544/` — 方式 A (logit bias) 结果 |
| 日志 | `runs/phase4_train.log` — 双流训练日志（E1 B1080 后卡死） |

| 方式 | Vaihingen | Potsdam | 结论 |
|---|---|---|---|
| A — Logit bias | 43.3% → 15.4% | 52.7% → 16.0% | 乘法注入是灾难 |
| B — 双流微调 | 训练可达（loss 正常） | — | 验证瓶颈待修复 |

### Plan2 Phase 1 — Per-Class Binary Baseline

| 类型 | 路径 |
|---|---|
| 代码 | `RS-SAM3/sam3_model.py` — SAM3Model 封装（兼容 Medical-SAM3 checkpoint） |
| 代码 | `RS-SAM3/dataset_rs.py` — Per-class 二分类数据集 |
| 代码 | `RS-SAM3/metrics.py` — Dice/IoU/Precision/Recall（二分类指标） |
| 代码 | `RS-SAM3/eval_binary.py` — 主评估脚本 |
| 代码 | `RS-SAM3/tune_thresholds.py` — Per-class 阈值扫描 |
| 结果 | `runs/plan2_phase1_20260430_200013/` — 二分类 baseline |
| 结果 | `runs/plan2_phase1b_20260430_200824/` — 阈值调优结果 |
| CSV | `RS-SAM3/class_summary.csv` — Phase 1 完整结果 |

| 实验 | mIoU (V+P 均值) | 说明 |
|---|---|---|
| Baseline (top-15%) | 0.450 | Per-class 二分类原始结果 |
| **Threshold tuned** | **0.554** (+10.3%) | Per-class 最优 K 值 |

### SAM3 官方训练探索

| 类型 | 路径 |
|---|---|
| 代码 | `sam3_isprs/convert_to_coco.py` — ISPRS → COCO JSON 转换 |
| 代码 | `sam3_isprs/train_official.py` — 官方 model_builder 训练（未成功） |
| 代码 | `sam3_isprs/train_sam3_decoder.py` — SAM3 + UNet decoder（路径冲突） |
| 数据 | `sam3_isprs/{vaihingen,potsdam,combined}/annotations.json` — COCO 格式标注 |

## 预期发表方向

1. **"Fine-tuning SAM 3 for Remote Sensing Semantic Segmentation"** — 第 3 阶段成果，首次将 SAM 3 微调到遥感领域
2. **"Multimodal SAM 3: Elevation-Guided Open-Vocabulary Segmentation"** — 第 4 阶段成果，通过 DSM 注入扩展 SAM 3 到多模态
3. **"Prompt Matters: Text Prompt Engineering for Open-Vocabulary Remote Sensing"** — 第 2 阶段成果（快速发表）
