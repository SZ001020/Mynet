# Plan4: SAM3 ViT 全量训练

> 日期：2026-05-07 | 继承 Plan3 最优架构（MFNet Decoder + DSM + 256² window）
> 目标：全量训练 SAM3 ViTDet backbone，复现 MFNet 的成功路径，突破 frozen 瓶颈（73% → 80-85%）

---

## 动机：渐进解冻失败，需要全量训练

### D1/D2 实验结果（已失败）

| 方法 | 可训练参数 | 256² mIoU | vs Frozen |
|------|-----------|-----------|-----------|
| Frozen (VPT+MFNet Decoder) | 4.5M | **73.10%** | — |
| D1 (unfreeze 8层, lr=1e-5) | 116M | 72.48% | -0.62 |
| D2 (unfreeze 16层, lr=1e-5) | 227M | 72.58% | -0.52 |

渐进解冻 + AdamW + 低学习率 = 在 3292 个样本上严重过拟合，epoch 2 就 peak。

### 为什么全量训练

MFNet 用 **SGD + lr=0.01 + momentum** 在 960 个 Vaihingen 样本上全量训练 SAM1 ViT-L（632M），拿到 85% mIoU。关键在于：

1. **SGD 的隐式正则化**：高学习率 SGD 的梯度噪声阻止大模型在小数据集上过拟合到 sharp minima
2. **从头调整特征空间**：456M 参数全部参与训练，ViT 的 attention pattern 可以直接适配遥感域，不需要 VPT 做中介
3. **已验证的 VRAM 可行性**：全量训练 SAM3 ViTDet (454M) + SGD 仅需 ~18GB (bs=1)，~30GB (bs=4)

---

## 训练配方（对标 MFNet）

| 配置 | MFNet | Plan4 |
|------|-------|-------|
| Backbone | SAM1 ViT-L (632M) | SAM3 ViTDet (454M) |
| 训练模式 | **全量训练** | **全量训练** |
| 优化器 | SGD momentum=0.9 | SGD momentum=0.9 |
| 学习率 | 0.01 | 0.01 → CosineAnnealing |
| Weight decay | 5e-4 | 5e-4 |
| Epoch | 50 | 50 |
| 窗口大小 | 256² | 256² |
| 训练样本 | 960 (Vaihingen) | 3292 (Vaihingen) |
| Batch size | 10 | 4 (单卡) / 8 (双卡 DDP) |
| Decoder | UNetFormer | MFNet Decoder (已验证) |
| DSM | ✓ | ✓ |
| 数据增强 | 旋转+翻转 | 旋转+翻转 |

---

## 实现计划

### 两张 5090 的使用

```
GPU 0: Vaihingen 全量训练 (bs=4, SGD, 50 epoch)
GPU 1: Potsdam 全量训练 (bs=4, SGD, 50 epoch)
```

两张卡独立训练，不通信。两个数据集分别拿到结果后，评估跨数据集泛化。

### 代码改动

基于 `RS-SAM3-p4/train_unfreeze.py` 修改：

1. **去掉 VPT Adapter**——全量训练不需要 VPT（ViT 自己学会了）
2. **所有 ViT 参数 requires_grad=True**
3. **换优化器**：AdamW → SGD(momentum=0.9, weight_decay=5e-4)
4. **加 warmup**：前 2 epoch 从 lr=1e-3 线性升到 lr=0.01
5. **CosineAnnealing**：lr 从 0.01 → 0 在 50 epoch 内
6. **梯度裁剪**：max_norm=1.0
7. **删除 language backbone**：节省 ~1.4GB VRAM

```python
# 关键代码
vb = model.backbone.vision_backbone
for p in vb.parameters():
    p.requires_grad = True  # 全量训练

# 删除不需要的组件释放 VRAM
del model.backbone.language_backbone

# SGD with momentum (matching MFNet)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
```

### 评估输出标准（所有 plan 统一）

所有 evaluate/validate 必须输出以下四项指标：

| 指标 | 说明 | JSON key |
|------|------|----------|
| Overall OA | 全部前景类总体准确率 | `avg_oa` |
| Overall mIoU | 全部前景类平均 IoU | `avg_miou` |
| Per-class IoU | 每个类别的 IoU | `per_class_iou` |
| Per-class OA | 每个类别的 OA (TP+TN)/total | `per_class_oa` |

Per-class OA 公式: `(inter[c] + total - union[c]) / total`（利用已有的 inter/union 直接推导，无需额外累加器）。

JSON 输出同时包含 per-class IoU 和 per-class OA，方便与 MFNet 论文的逐类别 OA 直接对比。

---

### 输出目录

```
GPU0: /root/autodl-tmp/runs/plan4_full_vaihingen_{ts}/
GPU1: /root/autodl-tmp/runs/plan4_full_potsdam_{ts}/
```

---

## 预期结果

| 数据集 | 预期 256² mIoU | 对标 |
|--------|---------------|------|
| Vaihingen | 78-83% | MFNet 85.03% |
| Potsdam | 80-85% | MFNet 86.69% |

即使达不到 MFNet 的水平（SAM1 plain ViT 架构更适合语义分割），全量训练 SAM3 ViTDet 也应该是我们最强的 baseline。

---

## 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| SGD 全量训练仍然过拟合 | 中 | 增加数据增强（ColorJitter）、加 dropout=0.2 |
| VRAM 不足（bs=4 全量训练） | 低 | bs=2 + gradient_accumulation=2 |
| 融合算子阻断梯度 | 极低 | ViT backbone 已验证无融合算子 |
| 训练 slow（~1.5h/ep × 50 = 75h） | 高 | 先跑 GPU0，GPU1 可并行跑其他实验 |

---

## 成功标准

- 256² mIoU ≥ **78%**（超过所有 frozen 实验 >5pp）
- 256² mIoU ≥ **80%**（论文可发水平）
- Potsdam 交叉验证（如果 GPU1 同时完成）
