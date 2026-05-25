# Plan5: SAM3 辅助边界/物体监督

> 日期：2026-05-08 | 启发：SAM_RS (IEEE TGRS 2024)
> 核心思路：不改架构，用 SAM3 zero-shot 产出边界图和物体实例图作为辅助监督

---

## 动机

SAM_RS 的核心 insight：**SAM 自己就是个优秀的边界/物体检测器**，在自然图像上预训练的 SAM 对物体边缘有极强的感知能力。把这个能力作为辅助训练信号，可以提升遥感域的分割边界质量。

```
SAM_RS 的 loss 设计:
  L = CE + λ₁·BoundaryLoss + λ₂·ObjectLoss
            ↑                    ↑
     SAM 提取的边界图       SAM 提取的物体连通分量图
```

我们的 structure_loss 已经做了边缘加权 BCE，但用的是 avg_pool 差分的粗糙边界。用 SAM3 zero-shot 主动提取的边界更精确，且物体级别的 Dice loss 对小目标（car）有帮助。

## 差距分析

| 指标 | 我们最优 | MFNet Frozen | 差距 |
|------|---------|-------------|------|
| Building OA | 90.5% | 94.6% | **-4.1**（边界不清） |
| Car OA | 76.4% | 76.8% | -0.4（小物体漏检） |
| Grass OA | 78.3% | 71.7% | +6.6（已领先） |

Building 的 -4.1pp 是最大弱点——SAM_RS 的 BoundaryLoss 正是针对这个问题设计的。

## 实现计划

### 步骤 1：SAM3 预处理（离线，~30min）

对 Vaihingen 12 张训练 tile，用 SAM3 zero-shot 推理：
- 输入：tile RGB 图
- 输出 1：**边界图**（SAM3 的 mask 边界 + sobel 边缘检测）
- 输出 2：**物体实例图**（SAM3 的 instance mask → 连通分量 ID）

```python
# 对每张训练 tile
for tile in train_tiles:
    # SAM3 zero-shot 推理
    masks = sam3_predict(tile)  # per-instance binary masks
    
    # 边界图: 所有 instance mask 边界的并集
    boundary_map = union([mask_to_boundary(m) for m in masks])
    
    # 物体实例图: 连通分量编号
    object_map = connected_components(boundary_map, gt_mask)
```

### 步骤 2：修改 loss（训练时）

在 `structure_loss` 基础上加两个辅助 loss：

```python
def combined_loss(pred_logits, gt_mask, boundary_map, object_map):
    # 1. 主 loss：structure_loss（边缘加权 BCE + IoU）
    loss_struct = structure_loss(pred_logits, gt_mask)
    
    # 2. 边界 loss：Dice loss on boundary pixels
    boundary_weight = 0.1
    pred_boundary = sobel_edge(pred_softmax)  # 预测的边界
    loss_boundary = dice_loss(pred_boundary, boundary_map) * boundary_weight
    
    # 3. 物体 loss：Dice loss on object instances
    object_weight = 0.5
    pred_objects = connected_components(pred_argmax)
    loss_object = multiclass_dice_loss(pred_softmax, object_map) * object_weight
    
    return loss_struct + loss_boundary + loss_object
```

### 步骤 3：训练

- 基于当前最优模型：VPT + MFNet Decoder + DSM + 256² windows
- 代码：`Personal-Project/RS-SAM3-p4/train_boundary.py`
- 配置：AdamW lr=1e-4, 20 epoch, batch=4
- 数据集：Vaihingen 256² windows（3292 samples）

## 预期收益

| 类别 | 当前 OA | 预期 OA | 收益来源 |
|------|--------|---------|---------|
| Building | 90.5% | 92-94% | BoundaryLoss 精确化边缘 |
| Car | 76.4% | 78-80% | ObjectLoss 改善小物体完整性 |
| Road | 89.2% | 90-91% | 边界改善 |
| Tree | 84.7% | 85-87% | 边界+物体 |
| Grass | 78.3% | 79-81% | 边界改善 |

预期整体 mIoU：73.5% → **75-77%**（+1.5-3.5pp）

## 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| SAM3 zero-shot 在遥感上质量差，边界图噪声大 | 中 | 用 GT mask 做 filtering，只保留与 GT 重叠的边界 |
| 物体 loss 与 structure_loss 冲突 | 低 | 用 warmup：前 5 epoch 不加辅助 loss |
| 预处理太慢（SAM3 推理 12 张图） | 低 | 12 张 tile × ~30s = 6min，可接受 |
| structure_loss 已经做了边缘加权，辅助 loss 冗余 | 中 | 对比实验：baseline vs +boundary vs +object vs +both |

## 评估

训练完成后用 `eval_universal.py --name "Plan5"` 评估 256² 全指标。

## 与 SAM_RS 的差异

| | SAM_RS | Plan5 |
|---|--------|-------|
| Backbone | SAM1 ViT-H 全量训练 | SAM3 ViTDet frozen |
| 主 Loss | CE | structure_loss（边缘加权 BCE+IoU） |
| 辅助 Loss | Boundary CE + Object Dice | **同**（借鉴） |
| 边界来源 | SAM1 预计算 | SAM3 zero-shot 预计算 |
| 预期 mIoU | 82.5% | 75-77%（frozen 限制） |
