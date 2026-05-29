# Phase 3: SAM 3 遥感微调 — Partial Fine-tuning 结果

> 实验时间: 2026-04-29 | GPU: RTX 5090 32GB | 策略: 冻结 SAM 3 backbone + 训练 FPN decoder
> 训练数据: Vaihingen 12 tiles + Potsdam 18 tiles = 30 tiles, 随机 512² crop
> Zero-shot baseline: Vaihingen 65.8%, Potsdam 55.6% (Semantic-Only, 5-class)

## 训练配置

| 参数 | 值 |
|---|---|
| 微调模式 | Partial (frozen backbone + trainable decoder) |
| 训练参数 | 797K / 841M (0.1%) |
| FPN decoder | 4-level lateral + smooth + 2-layer cls head (~0.8M params) |
| Epochs | 5 |
| Batch size | 2 |
| Crop size | 512×512 |
| 优化器 | AdamW, lr=1e-4, CosineAnnealing |
| 损失函数 | CrossEntropyLoss (ignore clutter pixels) |

## 训练曲线

| Epoch | Train Loss | Vaihingen mIoU | Potsdam mIoU |
|---|---|---|---|
| 1 | 0.670 | 47.1% | 21.3% |
| 2 | 0.523 | 47.4% | 22.0% |
| 3 | 0.453 | **52.1%** | 25.7% |
| 4 | 0.419 | 50.6% | 23.4% |
| 5 | 0.397 | 51.0% | 24.6% |

## 与 Zero-shot 对比

| 指标 | Zero-shot (Semantic-Only) | Fine-tuned (Partial) | Δ |
|---|---|---|---|
| Vaihingen 5c-mIoU | **65.8%** | 52.1% | **-13.7%** ❌ |
| Potsdam 5c-mIoU | **55.6%** | 25.7% | **-29.9%** ❌ |
| Vaihingen per-class (best) | [69.3, 86.3, 53.2, 68.3, 51.8] | [66.3, 71.1, 47.4, 48.5, 27.3] | — |

## 失败原因分析

### 1. 训练数据严重不足
仅 30 张遥感图像（Vaihingen 12 + Potsdam 18）用于训练语义分割模型，这是根本性限制。SAM 3 的 ViTDet backbone 输出 256 维特征，FPN decoder 需要足够多的样本才能学到有意义的类别边界。相比之下，zero-shot 利用 SAM 3 在海量数据上学到的 text-vision alignment，不需要遥感训练数据。

### 2. 特征-标签对齐问题
SAM 3 的 backbone 特征是为开放词汇检测（text-prompt driven）训练的，不是为逐像素分类优化的。直接在这些特征上训练 pixel-wise classifier 存在 domain gap。zero-shot 通过 text prompt 隐式地桥接了这个 gap，而 FPN decoder 没有这个桥梁。

### 3. 跨数据集泛化失败
Potsdam 的 25.7% vs Vaihingen 的 52.1% 差距极大（-26.4%），说明 FPN decoder 严重过拟合 Vaihingen 的数据分布。当训练集是混合数据集时，decoder 倾向于拟合样本更多的 Potsdam（18 tiles）中的简单模式，同时丢失 Vaihingen 的细节。

### 4. 类别不均衡
car 类别（27.3% in Vaihingen）的 IoU 远低于 building（71.1%），因为 car 是少数类（像素占比 <5%），CE loss 无法有效学习。

## 关键发现

1. **简单 FPN decoder 微调 远不如 zero-shot**：SAM 3 的零样本能力（65.8%）建立在强大的 text-vision alignment 上，用少量遥感数据训练轻量 decoder 反而破坏了这种能力。

2. **需要更好的微调策略**：
   - 使用 SAM 3 原生的 segmentation head（而非自定义 FPN）
   - 引入 text prompt 作为正则化（在训练时保留 text embedding 路径）
   - 使用更大的遥感数据集（如 LoveDA 的 2522 张训练图）
   - 使用 MFNet 式的 LoRA 注入在 backbone 内部进行 domain adaptation

3. **对于小数据集，zero-shot > fine-tuning**：这是一个重要的方法论文献——在遥感数据集上，SAM 3 的零样本能力超越了在小数据集上的 naive 微调。

## 后续改进方向

- 使用 SAM 3 的 native training pipeline (`sam3-main/sam3/train/`)
- 在 LoveDA (2522 张训练图) 上进行微调
- 使用 text prompt 保持 encoder 的开放词汇能力
- 引入 DSM 多模态信息 (Phase 4)

## 文件清单

```
/root/Mynet/autodl-tmp/runs/phase3_partial_20260429_231705/
├── best_model_vaihingen.pt       ← 最佳 Vaihingen checkpoint (epoch 3)
├── best_model_potsdam.pt         ← 最佳 Potsdam checkpoint (epoch 3)
├── training_history.json         ← 训练曲线数据
├── config.json                   ← 实验配置
└── phase3_results.md             ← 本报告
```
