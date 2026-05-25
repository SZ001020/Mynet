# Plan2: 基于 Medical-SAM3 启发的遥感分割二阶段计划

> 核心洞察来源: [Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3)
> SAM 系列的天然范式是 **prompt → binary mask**，不是 **image → class map**。
> 将遥感分割从多分类 argmax 切换为 per-class 二分类评估，能避开之前 Phase 1-4 的多个核心瓶颈。

> ⚠️ **评估协议**：最终评估必须使用 **256×256 滑动窗口 + overlap 平均**（与 MFNet/ISPRS 标准对齐）。Plan2 Phase 1 的 per-class 评估使用整图直接推理（非滑动窗口），结果仅作参考。后续所有最终结果需用 256² 协议重新评估。详见 plan3.md「评估协议」章节。
>
> ⚠️ **评估输出规范**：所有 evaluate/validate 必须输出四项指标：`avg_oa`, `avg_miou`, `per_class_iou`, `per_class_oa`。Per-class OA 公式：`(inter[c] + total - union[c]) / total`，可与 MFNet 论文逐类别 OA 直接对比。详见 plan3.md「评估输出规范」。

---

## 数据集划分

**所有实验统一使用 MFNet 标准划分（IEEE TGRS 2025），不可修改：**

```python
VAIHINGEN_TRAIN = ['1','3','23','26','7','11','13','28','17','32','34','37']  # 12 tiles
VAIHINGEN_TEST  = ['5','21','15','30']                                        # 4 tiles
POTSDAM_TRAIN   = ['6_10','7_10','2_12','3_11','2_10','7_8','5_10','3_12',
                   '5_12','7_11','7_9','6_9','7_7','6_8','4_12','6_12']     # 16 tiles
POTSDAM_TEST    = ['4_10','5_11','2_11','3_10','6_11','7_12']                 # 6 tiles
# Tile: Vaihingen=top_mosaic_09cm_area{id}, Potsdam=top_potsdam_{id}
```

## 总体思路

```
之前 (Phase 1-4):  image → 5 class logits → argmax → 5类分割图 → 混淆矩阵评估
                    ↑ 问题: clutter吸收、argmax竞争、多类CE loss、融合算子不可训

Plan2:             image → per-class text prompt → 5 binary masks → 独立二分类评估
                    ↑ 优势: 无类别竞争、可per-class阈值、对齐SAM3原生范式、微调可行
```

---

## 第 1 阶段：Per-Class 二分类基线重建

> **状态：✅ 已完成 (2026-04-30)**
> 代码: `/root/Mynet/RS-SAM3/`
> 结果: `/root/Mynet/autodl-tmp/runs/plan2_phase1_20260430_200013/`

### 1.1 Per-Class Binary IoU & Dice（5 类，无 clutter）

| 数据集 | 类别 | Dice | IoU | Precision | Recall | Plan1 多分类 IoU |
|---|---|---|---|---|---|---|
| Vaihingen | building | **0.777** | **0.648** | 0.964 | 0.667 | 86.1% |
| Vaihingen | car | 0.662 | 0.501 | 0.601 | 0.756 | 51.4% |
| Vaihingen | tree | 0.647 | 0.493 | 0.927 | 0.523 | 68.4% |
| Vaihingen | road | 0.592 | 0.423 | 0.957 | 0.433 | 69.7% |
| Vaihingen | grass | 0.399 | 0.255 | 0.740 | 0.309 | 50.0% |
| **Vaihingen 均值** | | **0.616** | **0.464** | | | **65.1%** |
| Potsdam | building | **0.752** | **0.615** | 0.950 | 0.640 | 81.9% |
| Potsdam | car | 0.744 | 0.592 | 0.704 | 0.793 | 54.3% |
| Potsdam | road | 0.581 | 0.412 | 0.871 | 0.444 | 61.7% |
| Potsdam | grass | 0.489 | 0.329 | 0.755 | 0.379 | 54.3% |
| Potsdam | tree | 0.373 | 0.235 | 0.898 | 0.242 | 33.0% |
| **Potsdam 均值** | | **0.588** | **0.437** | | | **57.1%** |

### 1.2 与 Plan1 多分类的对比

| 维度 | Plan1 多分类 IoU | Plan2 二分类 IoU | 说明 |
|---|---|---|---|
| building | 86.1% | 64.8% | 多分类中其他类"让出"了 building 像素 |
| car | 51.4% | 50.1% | 接近一致——car 是离散目标，二分类不受影响 |
| tree | 68.4% | 49.3% | 多分类从 grass/building 边界抢到了 tree 像素 |
| road | 69.7% | 42.3% | 多分类的 road 受益于排除 building/grass |
| grass | 50.0% | 25.5% | **最大差距**——grass 严重依赖多分类的"排除法" |
| Overall | 65.1% | 46.4% | 多分类 argmax 带来了 ~19% 的虚拟增益 |

### 1.3 关键发现

1. **Plan1 的多分类 IoU 高估了 SAM3 的真实 per-class 能力**——约 19% 的 IoU 差异来自 argmax 竞争而非更好的识别
2. **Building 在所有指标中都最强**（Dice=0.78, IoU=0.65）——SAM3 对建筑的识别极其精准（Precision=0.96），主要瓶颈是 recall（部分建筑未被检测）
3. **Road 的 precision 极高（0.96）但 recall 中等（0.43）**——SAM3 只标记它非常确定是道路的像素，遗漏了大量道路区域
4. **Grass 表现最差**（Dice=0.40, IoU=0.26）——Vegetation 类别的语义 head 激活模式分散，难以做清晰的二分类
5. **Precision-Recall trade-off 明显**：所有类的 precision >> recall，说明当前 top-15% 阈值偏保守。降低阈值可提升 recall 但会牺牲 precision

### 1.4 Phase 1b: Per-Class Threshold Tuning（零成本优化）

> **状态：✅ 已完成 (2026-04-30)**
> 结果: `/root/Mynet/autodl-tmp/runs/plan2_phase1b_20260430_200824/`

**方法：** 在 4 张校验图上扫描 top-K% 阈值 (K ∈ [3,5,8,10,12,15,18,20,25,30,35,40,50])，找到每类最优 K，应用到全量评估。Car 保持 baseline K=15（K=3 过拟合校验集）。

**最优阈值：**
| 类别 | Vaihingen K | Potsdam K | 说明 |
|---|---|---|---|
| road | 25% | 25% | 需要更大覆盖面 |
| building | 25% | 25% | 同上 |
| grass | 18% | 35% | Potsdam 草地更稀疏需更保守 |
| tree | 18% | 15% | 接近 baseline |
| car | **15%** | **15%** | 保持 baseline（小目标噪声大） |

**Tuned 结果 vs Baseline：**

| 类别 | V-Base | V-Tuned | Δ | P-Base | P-Tuned | Δ |
|---|---|---|---|---|---|---|
| road | 0.423 | **0.592** | +0.169 | 0.412 | **0.533** | +0.121 |
| building | 0.648 | **0.776** | +0.128 | 0.614 | **0.708** | +0.094 |
| grass | 0.255 | **0.417** | +0.162 | 0.329 | **0.421** | +0.092 |
| tree | 0.493 | **0.581** | +0.088 | 0.235 | **0.415** | +0.180 |
| car | 0.501 | 0.501 | ±0 | 0.592 | 0.592 | ±0 |
| **Overall** | **0.450** | **0.554** | **+0.103** | | | |

**关键发现：零训练成本，纯后处理阈值调优提升 10.3% mIoU。** SAM3 语义 head 的原始 logit 质量是好的，只是全局 top-15% 阈值过于保守。Per-class 自适应阈值释放了大量被过滤的有效像素。

### 1.5 代码文件

```
RS-SAM3/
├── sam3_model.py      ← SAM3Model 封装（兼容 Medical-SAM3 checkpoint 格式）
├── dataset_rs.py      ← Per-class 遥感数据加载器
├── metrics.py         ← Dice/IoU/Precision/Recall
├── eval_binary.py     ← 主评估脚本
└── class_summary.csv  ← Phase 1 完整结果

---

## 第 2 阶段：SAM3 遥感微调（中期，2-4周）

**目标：** 参考 Medical-SAM3 的 checkpoint 加载模式，实现遥感微调。

### 2.1 核心突破点

Medical-SAM3 证明了 **SAM3 可以被微调且微调后的权重可以加载**：

```python
# Medical-SAM3 的 checkpoint 加载方式
model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=None,    # 不加载预训练权重
    load_from_HF=False
)
# 加载微调后的自定义 checkpoint
ckpt = torch.load("medsam3_checkpoint.pt")
model.load_state_dict(clean_state_dict, strict=False)
```

这恰好避开了 Phase 3 的融合算子问题——**微调在外部完成（训练框架处理梯度），推理时只需加载 fp32 权重，不需要梯度。**

### 2.2 微调策略（三种）

| 策略 | 数据 | 方法 | 对标 |
|---|---|---|---|
| A. 医学预训练迁移 | 用 Medical-SAM3 checkpoint 直接推理遥感 | 零样本迁移 | 类似 Phase 1 但用微调过的权重 |
| B. 遥感微调（小数据集） | Vaihingen+Potsdam (30张) | LoRA 微调（在外部完成的 checkpoint） | 对标 Medical-SAM3 微调范式 |
| C. 遥感微调（大数据集） | LoveDA (2522张) | 全量/部分微调 | Phase 3 原本目标 |

### 2.3 微调数据格式

适配 Medical-SAM3 的二分类范式：

```python
# 每张图 × 每类 = 一个训练样本
@dataclass
class RSSample:
    image: np.ndarray          # RGB 遥感图 crop
    gt_mask: np.ndarray        # 该类的 binary mask
    text_prompt: str           # "building", "road", etc.
    bbox: Tuple               # 从 GT mask 生成的 bbox（可选）
```

这样一张 Vaihingen 图（5 类）= 5 个训练样本（每个类一个 binary mask）。

### 2.4 关键实验

1. **Medical-SAM3 zero-shot on RS**: 直接用医学微调过的 SAM3 推理遥感，看医学特征能否迁移
2. **Per-class fine-tune vs multi-class**: 二分类微调 vs 多分类微调的对比
3. **Box prompt vs Text prompt**: geometric prompt（从 GT bbox）vs text prompt 的效果差异

**预期产出：** 微调后的 per-class Dice/IoU + 与 zero-shot 的对比

---

## 第 3 阶段：Remote-SAM3 专用模型（长期，4-8周）

**目标：** 训练遥感专用的 "Remote-SAM3" checkpoint。

### 3.1 核心思路

```
Medical-SAM3 路径:  SAM3 → 医学数据微调 → MedSAM3 checkpoint
Remote-SAM3 路径:   SAM3 → 遥感数据微调 → RemoteSAM3 checkpoint
                                     ↑
                              可以加入 DSM 通道
```

### 3.2 三阶段数据策略

| 阶段 | 数据量 | 目标 |
|---|---|---|
| 3a. 概念验证 | Vaihingen 12 + Potsdam 18 = 30 张 | 验证微调 pipeline 可行 |
| 3b. 中等规模 | LoveDA 2522 张 | 建立 competitive baseline |
| 3c. 大规模 | 多个公开 RS 数据集合并 | 训练通用 Remote-SAM3 |

### 3.3 多模态扩展

在 Remote-SAM3 的基础上加入 DSM 通道：

```
方案 A — DSM 作为额外 geometric prompt（Phase 4 已验证不可行）
方案 B — 双流融合（Phase 4 尝试过，训练太慢）
方案 C — DSM → RGB 映射 + SAM3 原生推理 ✅
          ↑ 最实际: 用 DSM 增强 RGB 图像（如 hillshade 叠加），
            让 SAM3 "看到" 高程信息而无需改架构
```

方案 C 的具体做法：
```python
# 将 DSM 转为可视化特征叠加到 RGB 上
hillshade = compute_hillshade(dsm)  # 山体阴影
slope = compute_slope(dsm)          # 坡度
# 三通道合成: R=RGB_R, G=hillshade, B=slope
enhanced_rgb = np.stack([rgb[:,:,0], hillshade, slope], axis=-1)
# 直接喂入 SAM3（不需要改模型！）
```

### 3.4 评估体系

| 维度 | Phase 1（旧） | Plan2 第 3 阶段（新） |
|---|---|---|
| 指标 | IoU（混淆矩阵） | Per-class Dice + IoU（二分类） |
| 模型 | SAM3 zero-shot | Remote-SAM3 fine-tuned |
| 模态 | RGB only | RGB + DSM enhanced |
| 数据集 | Vaihingen/Potsdam | + LoveDA + iSAID + UAVid |
| 对比基线 | MFNet (SAM1) | Vanilla SAM3 + MedSAM3 + Remote-SAM3 |

---

## 时间线

| 周 | 阶段 | 关键产出 |
|---|---|---|
| 1-2 | 第 1 阶段：二分类基线重建 | Per-class binary Dice/IoU for Vaihingen+Potsdam |
| 3-4 | 第 2 阶段-A：Medical-SAM3 迁移 | MedSAM3 on RS 的 zero-shot 效果 |
| 5-6 | 第 2 阶段-B/C：遥感微调 | Fine-tuned checkpoint per-class 效果 |
| 7-8 | 第 3 阶段-a：概念验证 | Remote-SAM3 on ISPRS |
| 9-12 | 第 3 阶段-b/c + 多模态 | Remote-SAM3 + DSM on multi-dataset |

---

## 与 Plan1 的关系

| | Plan1（已完成） | Plan2（本文） |
|---|---|---|
| 核心范式 | 多分类语义分割 | Per-class 二分类 |
| Phase 1 对应 | Zero-shot 基线（已完成） | 二分类基线重建 |
| Phase 2 对应 | Prompt Engineering（已完成） | 微调 checkpoint 加载 |
| Phase 3 对应 | FPN decoder 微调（负结果） | SAM3 原生微调 |
| Phase 4 对应 | DSM logit bias（负结果） | DSM 可视化增强 |

> Plan1 的负结果（微调远不如 zero-shot、DSM 先验有害）恰好支持了 Plan2 的方向：
> SAM3 不应该被扭成多分类模型，而应该保持其原生二分类范式。

## 实验文件清单

### Plan2 Phase 1 — Per-Class Binary Baseline

| 类型 | 路径 |
|---|---|
| 代码 | `RS-SAM3/sam3_model.py` — SAM3Model 封装 |
| 代码 | `RS-SAM3/dataset_rs.py` — Per-class 二分类数据集 |
| 代码 | `RS-SAM3/metrics.py` — Dice/IoU/Precision/Recall |
| 代码 | `RS-SAM3/eval_binary.py` — 主评估脚本 |
| 结果 | `runs/plan2_phase1_20260430_200013/` — Baseline (per-class IOU) |
| CSV | `RS-SAM3/class_summary.csv` — 完整结果 |

| 类别 | Vaihingen IoU | Potsdam IoU |
|---|---|---|
| building | **0.648** | **0.614** |
| car | 0.501 | 0.592 |
| tree | 0.493 | 0.235 |
| road | 0.423 | 0.412 |
| grass | 0.255 | 0.329 |
| **Overall** | **0.450** | **0.437** |

### Plan2 Phase 1b — Per-Class Threshold Tuning

| 类型 | 路径 |
|---|---|
| 代码 | `RS-SAM3/tune_thresholds.py` — Top-K 阈值扫描 + 全量评估 |
| 结果 | `runs/plan2_phase1b_20260430_200824/` — 校准曲线 + tuned 结果 |
| CSV | `runs/plan2_phase1b_20260430_200824/tuned_summary.csv` |

| 类别 | Best K (V) | Best K (P) | Baseline IoU | Tuned IoU | Δ |
|---|---|---|---|---|---|
| road | 25% | 25% | 0.423 | **0.592** | +0.169 |
| building | 25% | 25% | 0.648 | **0.776** | +0.128 |
| grass | 18% | 35% | 0.255 | **0.417** | +0.162 |
| tree | 18% | 15% | 0.493 | **0.581** | +0.088 |
| car | 15% | 15% | 0.501 | 0.501 | ±0 |
| **Overall** | | | **0.450** | **0.554** | **+0.103** |
