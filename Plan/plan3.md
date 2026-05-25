# Plan3: SAM3 适配遥感分割的三条路线

> 启发来源: [SAM3-UNet](https://github.com/) 的 Adapter+VPT 模式和 [SAM3_LoRA](https://github.com/) 的 LoRA 注入模式
> 背景: Plan1 的多分类微调劣于 zero-shot（Phase 3），Plan2 确定了 per-class 二分类范式
> 核心目标: 在 per-class binary 范式下，找到能超过 zero-shot baseline 的微调策略

---

## 评估协议（统一标准）

**最终评估必须使用 256×256 滑动窗口 + overlap 平均（与 MFNet 论文协议一致）。**

| 协议 | 用途 | 说明 |
|------|------|------|
| 512² crop 评估 | **训练监控**（快速参考） | 会系统性低估性能，不可作为最终指标 |
| 1008² 滑动窗口 | 中间参考 | 接近模型原生分辨率，指标偏高 |
| **256² 滑动窗口** | **最终评估（强制）** | stride=128, overlap=50%, 与 MFNet 论文对齐 |

**原因**（已验证）：
- 512² crop 评估 vs 256² 滑动窗口差距：Vaihingen 67.1%→68.7%（+1.6pp），Potsdam 47.2%→74.0%（+26.8pp）
- 1008² 窗口高估了性能（76-77%），因为模型在该分辨率训练
- 256² 窗口最接近真实部署场景和学术界基准

### 评估输出规范（所有 evaluate/validate 必须遵守）

每个评估函数必须输出 **四项指标**：

| 指标 | JSON key | 说明 |
|------|----------|------|
| Overall OA | `avg_oa` | 全部前景类总体准确率 |
| Overall mIoU | `avg_miou` | 全部前景类平均 IoU |
| Per-class IoU | `per_class_iou` | 每个类别的 IoU（字典） |
| Per-class OA | `per_class_oa` | 每个类别的 OA = (TP+TN)/total（字典） |

Per-class OA 公式（利用已有的 inter/union 直接推导，无需额外累加器）：
```
pc_oa[c] = (inter[c] + total - union[c]) / total * 100
```
推导：TN = total - TP - FP - FN = total - union[c]，OA = (TP + TN) / total = (inter[c] + total - union[c]) / total。

这样可以与 MFNet 论文的逐类别 OA 直接对比（MFNet Table I 报告的是 per-class OA，不是 per-class IoU）。

---

## 关键瓶颈回顾

Plan1 Phase 3 失败的两个根因：

1. **FPN decoder 太弱**（797K 可训练参数，无法学习 text-vision alignment）→ zero-shot 65.8% vs fine-tuned 56.6%
2. **SAM3 融合算子不支持 autograd**（`perflib/fused.py`、`vlcombiner.py` 中的自定义 CUDA kernel 阻断梯度流）

Plan3 三条路线的设计原则：
- **不修改 SAM3 内部的融合算子**，它们保持 frozen 和 no_grad
- **在标准 PyTorch 层上注入可训练参数**（Adapter 在 feature 空间，LoRA 在 Linear 层）
- **维持 per-class binary 范式**（Plan2 已验证的路线）

---

## 路线 A — Adapter + UNet Decoder（低风险、快速验证）

```
┌─────────────────────────────────────────────────────────┐
│ SAM3 ViT backbone (frozen, no_grad)                      │
│   Block 0 → Adapter 0 → Block 1 → Adapter 1 → ...      │
│          ↓               ↓                    ↓          │
│      [feature maps at 4 scales from intermediate layers] │
├─────────────────────────────────────────────────────────┤
│ UNet Decoder (trainable)                                 │
│   ← 4-level upsampling with skip connections             │
│   ← LightBlock or ConvBlock per level                    │
│   → 5-channel output (per-class logits)                  │
├─────────────────────────────────────────────────────────┤
│ Loss: structure_loss = 边缘加权 BCE + 加权 IoU           │
│   - 边界像素 weight = 1 + 5×|avg_pool(mask) - mask|     │
│   - 对 building/road 边界精度提升关键                     │
└─────────────────────────────────────────────────────────┘
```

### 核心组件

**Adapter（Visual Prompt Tuning 变体）:**
```python
class Adapter(nn.Module):
    def __init__(self, blk):
        self.block = blk           # 冻结的 ViT block
        dim = blk.attn.qkv.in_features  # 1024
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),    # 1024 → 32（压缩比 32:1）
            nn.GELU(),
            nn.Linear(32, dim),    # 32 → 1024
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)   # 学习到的特征扰动
        return self.block(x + prompt)   # 残差注入到冻结 block
```

每个 Adapter ~65K 参数，32 层总计 ~2M 可训练参数。

**UNet Decoder（可复用 fineNet/ 中的现有实现）:**
- `Reference-Project/fineNet/train_phase3_unet.py` 的 UNet decoder
- `Reference-Project/fineNet/train_dual_stream.py` 的 ConvBlock + Up 模块
- 输出改为 5-channel（per-class logits），不做 argmax

**Loss:**
```python
def structure_loss(pred_logits, gt_mask):
    # 边缘加权：边界像素 ×5 权重
    weit = 1 + 5 * torch.abs(F.avg_pool2d(gt_mask, 31, 1, 15) - gt_mask)
    wbce = (weit * F.binary_cross_entropy_with_logits(pred_logits, gt_mask)).sum() / weit.sum()
    # 加权 IoU
    pred = torch.sigmoid(pred_logits)
    inter = ((pred * gt_mask) * weit).sum()
    union = ((pred + gt_mask) * weit).sum()
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
```

### 优势
- **零融合算子风险**：只用 ViT backbone（纯标准 PyTorch），完全避开 `perflib/fused.py`
- **参数量适中**：~2M（vs Plan1 FPN decoder 的 797K），3× 容量提升
- **可快速验证**：复用 `Reference-Project/fineNet/` 现有 decoder 代码，改动量最小
- **SAM3-UNet 已验证可行**（在 saliency detection 上 work）

### 劣势
- **丢失 text prompt 能力**：不用 SAM3 的 text encoder，退化为纯视觉模型
- **多分类仍然需要 argmax**：5-channel 输出 → argmax，但可以用 per-class Dice 评估
- **无开放词汇能力**：看不到的类别无法推理

### 数据配置
- 训练: Vaihingen 12 tiles 或 Potsdam 18 tiles，crop 到 512² patches
- 验证: Vaihingen 4 test tiles + Potsdam 6 test tiles，per-class Dice/IoU
- 学习率: 1e-4 (AdamW)，CosineAnnealing，50 epochs
- Batch size: 8-16 (32GB VRAM)

### 成功标准
- Vaihingen per-class mIoU > 65.8%（超过 zero-shot baseline）
- Potsdam per-class mIoU > 56.6%（超过 zero-shot baseline）

### 实验结果（2026-05-01/02）

**代码：** `Personal-Project/RS-SAM3-p3/` | **数据划分：** MFNet splits (gts_for_participants) | **评估协议：** 256² 滑动窗口

#### Vaihingen（NIRRG, 12 train / 4 val）

| 方法 | OA | mIoU | road | building | grass | tree | car |
|------|-----|------|------|----------|-------|------|-----|
| Route A (RGB) | 84.2% | 68.7% | 71.3 | 81.6 | 51.9 | 73.8 | 64.9 |
| Route A+DSM (RGB+DSM) | **84.7%** | **70.0%** | **72.8** | 81.6 | **55.9** | **74.5** | **65.5** |
| *DSM Δ* | *+0.5* | *+1.3* | *+1.5* | *0.0* | *+4.0* | *+0.7* | *+0.6* |

**✅ Route A 成功：超过 Plan1 zero-shot baseline (65.8%) +2.9pp**
**✅ Route A+DSM 有效：进一步 +1.3pp (70.0%)，主要帮助 grass (+4pp) 和 car**

训练配置：batch=4, lr=1e-4, weight_decay=1e-3, Dropout2d=0.1, 20 epochs

#### Potsdam（RGBIR, 18 train / 6 val）

| 方法 | OA | mIoU | road | building | grass | tree | car |
|------|-----|------|------|----------|-------|------|-----|
| Route A (RGB) | 88.0% | 74.0% | 75.8 | 83.7 | 62.8 | 69.3 | 78.5 |

**✅ 超过 Plan1 zero-shot (56.6%) +17.4pp**。Potsdam (18 tiles) 泛化优于 Vaihingen (12 tiles)。

> ⚡ **与 MFNet 对比**：我们的 sota（Vaihingen 70.0%, Potsdam 74.0%）距 MFNet ViT-H+MMAdapter（85.0%, 86.7%）差 15-17pp mIoU。主要原因：(1) MFNet 全量训练 SAM1，我们 frozen SAM3 + adapter；(2) MFNet 用 UNetFormer decoder（GLA blocks），我们用简单 UNet；(3) MFNet 用 DSM 双流 encoder 内部融合，我们做 post-hoc concat。

#### 关键发现
1. **256² 滑动窗口评估是唯一可靠指标**：512² crop 严重低估（Potsdam 47%→74%，差 27pp），1008² 高估。256² 与 MFNet 对齐。
2. **DSM 在所有协议下均正向**：+1.1~1.3pp mIoU，对 grass/car 等依赖高程的类别尤其有效
3. **Adapter-VPT 有效**：~6.6M 可训练参数超越 zero-shot，纯视觉模型无需 text prompt
4. **训练 tile 数量影响大**：Potsdam (18) OA/mIoU 反超 Vaihingen (12)，更多数据 > NIR 通道优势

### 全版本多维度对比 (Vaihingen, 256² 协议, 2026-05-04)

#### 参数与架构

| 模型 | PEFT | PEFT参 | Decoder | DSM编码 | DSM融合 | 总参数 | Batch |
|------|------|--------|---------|---------|---------|--------|-------|
| Route A RGB | VPT | 2.1M | 4.4M | — | — | 6.6M | 4 |
| Route A+DSM concat | VPT | 2.1M | 4.4M | CNN 57K | concat+1×1 | 6.9M | 4 |
| Route A+DSM cross-attn | VPT | 2.1M | 6.5M | CNN 684K | cross-attn | 9.8M | 4 |
| Route B LoRA | r=8 | 4.5M | 6.5M | — | — | 11.0M | 4 |
| Route B LoRA+DSM | r=8 | 4.5M | 6.5M | CNN 684K | gated conv | 17.3M | 2 |
| Route B MFNet-style | r=8 | 4.5M | 6.7M | SAM3 ViT | 4×SEFusion | 13.3M | 2 |

#### 训练效率

| 模型 | Ep | 最优Ep | 批/ep | 总步数 | 时长 | 耗时/ep | Forward |
|------|-----|--------|-------|--------|------|---------|---------|
| Route A RGB | 20 | 7 | 240 | 4,800 | 1.9h | 337s | 1× SAM3 |
| Route A+DSM concat | 20 | 6 | 240 | 4,800 | 1.9h | 346s | 1× SAM3 |
| Route A+DSM cross-attn | 20 | 3 | 240 | 4,800 | 2.1h | 372s | 1× SAM3 |
| Route B LoRA | 20 | 3 | 240 | 4,800 | 2.5h | 443s | 1× SAM3 |
| Route B LoRA+DSM | 20 | 3 | 480 | 9,600 | 2.5h | 455s | 1× SAM3 |
| Route B MFNet-style | 20 | 9 | 480 | 9,600 | 5.3h | 960s | 2× SAM3 |

#### 最终性能 (256² 协议)

| 模型 | 256² mIoU | Crop Best | Δ | road | building | grass | tree | car |
|------|-----------|-----------|-----|------|----------|-------|------|-----|
| Route A RGB | 68.70% | 67.1% | +1.6 | 71.3 | 81.6 | 51.9 | 73.8 | 64.9 |
| Route A+DSM concat | 70.04% | 63.9% | +6.1 | 72.8 | 81.6 | 55.9 | 74.5 | 65.5 |
| **Route A+DSM cross-attn** | **70.39%** | 66.0% | +4.4 | 73.4 | 83.0 | 56.2 | 74.5 | 64.8 |
| Route B LoRA | 69.70% | 66.8% | +2.9 | 72.3 | 80.9 | 56.6 | 74.1 | 64.6 |
| Route B LoRA+DSM | 69.63% | 65.0% | +4.6 | 72.9 | 82.6 | 55.4 | 74.1 | 63.2 |
| Route B MFNet-style | 70.19% | 64.0% | +6.2 | 73.4 | 83.0 | 56.9 | 74.9 | 62.8 |

#### 总结结论

1. **最高 mIoU**：Route A VPT + cross-attn DSM = **70.39%** (9.8M, 2.1h)
2. **最佳性价比**：Route A VPT + concat DSM = 70.04% (6.9M, 1.9h)
3. **VPT > LoRA**：~2M 的 VPT Adapter 在 frozen SAM3 上超过 ~4.5M 的 LoRA——token-space 扰动比 attention 低秩分解更适配遥感域
4. **DSM 增益 ~2pp**：RGB-only 68.7% → RGB+DSM 70.4%，DSM 在各融合方式下均正向但增益有限
5. **所有模型存在过拟合**：crop peak 在 epoch 3-9，256² 比 crop 高 1.6-6.2pp
6. **MFNet(85%) vs Ours(70%) 差距 ~15pp**：核心原因 = SAM1 全量训练 vs SAM3 frozen，非 DSM 融合方式
7. **512² crop 评估不可靠**：系统性地低估全图性能（差距 1.6-6.2pp），256² 滑动窗口为唯一可靠协议

---

## 路线 B — LoRA 微调 SAM3 ViT（已完成, 2026-05-03~04）

```
┌─────────────────────────────────────────────────────────┐
│ Full SAM3 (frozen backbone)                              │
│                                                          │
│  Vision Encoder (ViT, 32 layers)                         │
│    └── LoRA on Q/K/V/out_proj of each attention layer    │
│  Text Encoder (CLIP-style)                               │
│    └── LoRA on c_fc/c_proj of each MLP                   │
│  DETR Encoder (6 layers, cross-attention)                │
│    └── LoRA on Q/K/V/out_proj                             │
│  DETR Decoder (6 layers, 200 object queries)             │
│    └── LoRA on Q/K/V/out_proj                             │
│  Mask Decoder (3 stages, frozen)                         │
│    └── 保持通用 mask 解码能力                              │
├─────────────────────────────────────────────────────────┤
│ 每类独立推理: text_prompt → binary mask → Dice/IoU       │
└─────────────────────────────────────────────────────────┘
```

### 实际架构 (与原始计划不同)

> SAM3 fused ops 阻断梯度 + MHA 替换复杂度过高。当前为 **LoRA on ViT only + UNetFormer decoder**（Hybrid）。

```
SAM3 ViT (frozen) + LoRA on QKV/proj/MLP (128 layers, r=8, 4.5M)
  → FPN → UNetFormer Decoder (GLA, 6.5M) → 5-class argmax
```

### 实现变体 (代码: RS-SAM-p3b/)

| 变体 | 文件 | 描述 |
|------|------|------|
| B-LoRA | lora_sam3.py | LoRA ViT + UNetFormer, RGB only |
| B-LoRA+DSM | train_lora.py --dsm | +CNN DSM encoder + gated conv fusion |
| B-MFNet | lora_mfnet.py | DSM→3ch→共享SAM3 ViT + SEFusion + 4-scale |

### 实验结果 (Vaihingen, 256²)

| 变体 | mIoU | 参数 | 时间 | vs A最优(70.39) |
|------|------|------|------|-----------------|
| B-LoRA | 69.70% | 11.0M | 2.5h | -0.69pp |
| B-LoRA+DSM | 69.63% | 17.3M | 2.5h | -0.76pp |
| B-MFNet | 70.19% | 13.3M | 5.3h | -0.20pp |

### B 路线关键发现
1. LoRA 未超越 VPT：4.5M > 2.1M 但 69.70 < 70.39
2. DSM 对 LoRA 无效：LoRA+DSM (69.63) < LoRA-only (69.70)
3. MFNet-style 最贵但未最优：双 SAM3 forward + 4-scale = 70.19%
4. 完整 Route B (LoRA on DETR/text/mask) 未实现——需替换 MHA 模块

---

## 路线 C — 两阶段渐进（长期目标）

```
阶段 1 (路线 A)                  阶段 2 (路线 B)
┌──────────────────┐           ┌──────────────────┐
│ Adapter + UNet   │           │ Full SAM3 + LoRA │
│                  │           │                  │
│ 验证:             │  确认     │ 验证:             │
│ - 无融合算子问题  │ ──────▶  │ - text alignment │
│ - decoder 容量够 │  有效     │ - 跨数据集泛化   │
│ - structure_loss │           │ - 开放词汇能力   │
│   边缘权重有效   │           │                  │
│                  │           │                  │
│ 目标: > zero-shot│           │ 目标: > MFNet    │
│ 时间: 1-2 周     │           │ 时间: 3-4 周     │
└──────────────────┘           └──────────────────┘
```

### 阶段 1：Adapter 快速验证（1-2 周）
1. 在 `Reference-Project/fineNet/` 中实现 `AdapterViT` 包装类
2. 接入现有 UNet decoder（`train_phase3_unet.py` 或 `train_dual_stream.py`）
3. 替换 CE loss 为 `structure_loss`
4. 在 Vaihingen+Potsdam 上训练，per-class binary 评估
5. 关键实验：Adapter 的 rank（32/64/128）、注入层数（全部 32 层 vs 最后 8 层）、是否加 DSM 分支

### 阶段 2：LoRA 完整微调（3-4 周）
1. 使用 `SAM3_LoRA-main/lora_layers.py` 的 LoRA 实现
2. 从 Medical-SAM3 checkpoint 或 HF `facebook/sam3` 加载权重
3. 按 B1→B2→B3 渐进开放 LoRA 组件
4. 在 ISPRS (V+P) 验证，再扩展到 LoveDA
5. 与 Plan2 Phase 1 tuned baseline 对比

### 两个阶段的衔接条件

- **阶段 1 → 阶段 2 的门槛**：路线 A 的 mIoU > 65.8%（Vaihingen zero-shot）
  - ✅ **已达**：Route A 256² mIoU=70.4%，远超市值。Decoder+Adapter 范式验证通过
  - 路线 B (Hybrid LoRA on ViT) 已实现，但未超越 A（69.7% vs 70.4%）
  - 完整 Route B (LoRA on DETR/text/mask decoder) 待实现，需解决 MHA 替换问题

### 当前最优与下一步

- **当前最优**：Route A VPT Adapter + cross-attn DSM = 70.39% (256²)
- **核心瓶颈**：SAM3 frozen（~814M 参数冻结，仅训练 ~10M） vs MFNet 全量训练 SAM1
- **最大未验证方向**：LoRA on SAM3 DETR/text/mask decoder（真正的 Route B）
- **DSM 边际价值已近上限**：从 RGB-only 68.7% → best DSM 70.4%，+1.7pp

---

## 三条路线对比

| 维度 | 路线 A: Adapter+UNet | 路线 B: LoRA-SAM3 | 路线 C: 两阶段 |
|------|---------------------|----------------------|---------------|
| 状态 | ✅ 完成 | ⚠️ Hybrid 完成 | ⏳ A→B 衔接中 |
| 使用 SAM3 组件 | 仅 ViT backbone | ViT (Hybrid) / Full (未实现) | A 验证 → B 完善 |
| 可训练参数 | 6.6~9.8M | 11.0~17.3M | A:9.8M → B:TBD |
| Text prompt | 无（纯视觉） | 无（Hybrid）/ 有（Full） | 最终有 |
| 融合算子风险 | 零（只用 ViT） | 零（LoRA 在 nn.Linear） | 零 |
| 实际 256² mIoU | **70.39%** | 69.70% (Hybrid) | — |
| 训练时间 | 1.9~2.1h | 2.5~5.3h | — |
| 最大优势 | 最优性能+最快训练 | 保留 SAM3 全部能力(未实现) | 渐进风险控制 |
| 最大劣势 | 丢失 text prompt | 工程复杂度高 | 时间成本最高 |
| 预期时间 | 1-2 周 | 3-4 周 | 4-6 周 |

---

## 文件结构规划

```
fineNet/
├── adapter_vit.py          ← 新增: Adapter + ViT backbone 封装
├── adapter_unet.py         ← 新增: Adapter-ViT + UNet decoder 完整模型
├── train_adapter.py        ← 新增: 路线 A 训练脚本
├── lora_sam3_model.py      ← 新增: 路线 B LoRA SAM3 封装
├── train_lora.py           ← 新增: 路线 B 训练脚本
├── structure_loss.py       ← 新增: 边缘加权 BCE + IoU loss
├── train_dual_stream.py    ← 已有: 可复用 ConvBlock/DSMEncoder
├── train_phase3_unet.py    ← 已有: 可复用 UNet decoder
└── finetune_dataset.py     ← 已有: 可复用数据加载

RS-SAM3/
├── eval_binary.py          ← 已有: 路线 A/B 的评估脚本
├── tune_thresholds.py      ← 已有: per-class 阈值调优
└── dataset_rs.py           ← 已有: per-class 数据加载

SAM3_LoRA-main/
├── lora_layers.py          ← 参考: LoRA 层定义（可直接复用）
├── train_sam3_lora.py      ← 参考: 训练框架
└── configs/                ← 参考: LoRA 配置模板
```

---

## 决策树

```
Q: 你是不是想最快验证"微调能超过 zero-shot"？
├── 是 → 路线 A（Adapter+UNet）。1 周内出结果。
│
Q: 你是否需要保留开放词汇推理能力（新类别 zero-shot）？
├── 是 → 路线 B（LoRA-Full-SAM3）。保留 text prompt。
│
Q: 你是不是要做完整的研究论文（对标 MFNet 75%+ mIoU）？
└── 是 → 路线 C（两阶段）。先 A 验证范式，再 B 完善能力。
```

---

## 与 Plan1/Plan2 的关系

| | Plan1 | Plan2 | Plan3 |
|---|---|---|---|
| 范式 | 多分类 argmax | Per-class binary | Per-class binary |
| SAM3 使用 | 完整（zero-shot）→ 部分（FPN训练） | 完整（zero-shot） | Adapter-ViT 或 LoRA-SAM3 |
| 训练 | FPN decoder（负结果） | 无训练（阈值调优） | Adapter / LoRA PEFT |
| DSM | Logit bias（负结果） | Hillshade 可视化（待验证） | 路线 A 可复现 dual-stream DSM |
| 状态 | ✅ 已完成 | ✅ Phase 1 完成 | 📋 本文档 |
