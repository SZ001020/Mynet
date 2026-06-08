# MFNet 标准评估协议

> 本文档定义与 MFNet 论文 (IEEE TGRS 2025) 严格对标的评估标准。
> 所有声称 "与 MFNet 论文可对比" 的实验必须遵循此协议。
>
> 最后更新: 2026-06-06

---

## 1. 协议参数

| 参数 | Vaihingen | Potsdam | 说明 |
|------|:---:|:---:|------|
| 窗口大小 | 256×256 | 256×256 | `WINDOW_SIZE = (256, 256)` |
| **步长 (stride)** | **32** | **32** | 硬编码在 `train.py` MODE=='Test' 分支 (line 532, 543) |
| 边缘裁剪 (trim) | **0** | **0** | 不做边缘裁剪；边缘窗口向内移动对齐边界 |
| 标签 | **eroded (noBoundary)** | **eroded (noBoundary)** | ISPRS 标准侵蚀标签 |
| Logit 累积 | **soft-logit** | **soft-logit** | 跨窗口累积 logits，最后 argmax |
| DSM 归一化 | per-tile min-max | per-tile min-max | `(dsm - min) / (max - min)` |
| RGB 归一化 | `/255.0` | `/255.0` | 全局归一化 |
| 指标计算 | `np.nanmean(MIoU[:5])` | `np.nanmean(MIoU[:5])` | 前 5 类 (排除 clutter) |
| Per-class OA | Recall = TP/(TP+FN) | Recall = TP/(TP+FN) | MFNet 论文的 "per-class OA" 实际是 Recall |

### 关键代码溯源

**stride=32 硬编码** (`Reference-Project/MFNet/train.py`):
```python
elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    elif DATASET == 'Potsdam':
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
```

**边缘处理** (`Reference-Project/MFNet/utils.py`, `sliding_window` 函数):
```python
def sliding_window(top, step=10, window_size=(20,20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]  # 最后一行向内移动
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]  # 最后一列向内移动
            yield x, y, window_size[0], window_size[1]
```

**Soft-logit 累积** (`Reference-Project/MFNet/train.py`, `test` 函数):
```python
pred = np.zeros(img.shape[:2] + (N_CLASSES,))   # (H, W, 6)
for coords in grouper(batch_size, sliding_window(...)):
    outs = net(image_patches, dsm_patches, mode='Test')
    for out, (x, y, w, h) in zip(outs, coords):
        pred[x:x+w, y:y+h] += out.transpose((1,2,0))  # 累积 logits
pred = np.argmax(pred, axis=-1)  # 最后 argmax
```

**批量推理** (`Reference-Project/MFNet/train.py`):
```python
image_patches = np.asarray(image_patches)  # 堆叠为 batch
image_patches = torch.from_numpy(image_patches).cuda()
outs = net(image_patches, dsm_patches, mode='Test')  # 一次 forward
```

---

## 2. 数据集

### 2.1 Vaihingen

| 项目 | 值 |
|------|-----|
| 训练 tile | 1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37 (12 张) |
| 测试 tile | 5, 21, 15, 30 (4 张) |
| 图像路径 | `top/top_mosaic_09cm_area{N}.tif` |
| 标签路径 | `gts_for_participants/top_mosaic_09cm_area{N}.tif` (训练) |
| 侵蚀标签 | `gts_eroded_for_participants/top_mosaic_09cm_area{N}_noBoundary.tif` (评估) |
| DSM 路径 | `dsm/dsm_09cm_matching_area{N}.tif` |
| 通道 | NIRRG (3 波段: NIR + Red + Green) |
| GSD | 9 cm |
| 类别 | 6 (5 前景 + clutter, clutter 在评估中排除) |

### 2.2 Potsdam

| 项目 | 值 |
|------|-----|
| 训练 tile | 6_10, 7_10, 2_12, 3_11, 2_10, 7_8, 5_10, 3_12, 5_12, 7_11, 7_9, 6_9, 7_7, 4_12, 6_8, 6_12, 6_7, 4_11 (18 张) |
| 测试 tile | 4_10, 5_11, 2_11, 3_10, 6_11, 7_12 (6 张) |
| 图像路径 | `2_Ortho_RGB/top_potsdam_{tile}_RGB.tif` |
| 标签路径 | `5_Labels_for_participants/top_potsdam_{tile}_label.tif` (训练) |
| 侵蚀标签 | `5_Labels_for_participants_no_Boundary/top_potsdam_{tile}_label_noBoundary.tif` (评估) |
| DSM 路径 | `1_DSM/dsm_potsdam_{tile}.tif` |
| 通道 | RGB (3 波段) |
| GSD | 5 cm |
| 类别 | 6 (5 前景 + clutter, clutter 在评估中排除) |

> **注意**: MFNet 原代码从 `4_Ortho_RGBIR/` 加载 4 波段 TIFF 并取前 3 通道 `[:,:,:3]`。经验证，这与 `2_Ortho_RGB/` 的 3 波段 TIFF 逐像素完全一致。两者可互换使用。

### 2.3 类别映射

| ISPRS 类别 | MFNet 论文名称 | 注册表名称 | 索引 | 评估 |
|------------|:---:|------|:---:|:---:|
| Impervious surfaces | Imp. / roads | road | 0 | ✅ |
| Building | Bui. / buildings | building | 1 | ✅ |
| Low vegetation | Low. / low veg. | grass | 2 | ✅ |
| Tree | Tre. / trees | tree | 3 | ✅ |
| Car | Car / cars | car | 4 | ✅ |
| Clutter/background | clutter | — | 5 | ❌ 排除 |

颜色调色板:
```
0: (255, 255, 255)  — Impervious surfaces (白)
1: (0, 0, 255)      — Buildings (蓝)
2: (0, 255, 255)    — Low vegetation (青)
3: (0, 255, 0)      — Trees (绿)
4: (255, 255, 0)    — Cars (黄)
5: (255, 0, 0)      — Clutter (红, 评估中忽略)
```

---

## 3. 指标计算

### 3.1 从混淆矩阵推导

```python
import numpy as np

# confusion_matrix: shape [6, 6] 或 [5, 5], dtype=int64
# 行 = GT, 列 = Prediction
# 仅取前 5 类 (排除 clutter)

cm = confusion_matrix[:5, :5]

# Per-class IoU
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
per_class_iou = tp / (tp + fp + fn)

# Per-class Recall (= MFNet 论文的 "per-class OA")
per_class_recall = tp / (tp + fn)

# Per-class Precision
per_class_precision = tp / (tp + fp)

# Mean IoU (前 5 类)
miou = np.nanmean(per_class_iou)

# Overall Accuracy
oa = tp.sum() / cm.sum()

# Mean Recall
mean_recall = np.nanmean(per_class_recall)
```

### 3.2 MFNet 论文 "per-class OA" 歧义

**重要**: MFNet 论文中的 "per-class OA" 列实际是 **per-class Recall**。
- 它计算的是 `TP_c / (TP_c + FN_c)`，即"该类 GT 像素中有多少被正确预测"
- **不是** 传统意义上的 per-class OA = `(TP_c + TN_c) / Total`
- 两者在 car, grass 这类像素占比小的类别上差异巨大（TN 会拉高 OA）
- 与 MFNet 论文对比时，**必须使用 Recall**，禁止使用含 TN 的公式

---

## 4. 评估脚本实现检查清单

编写符合此协议的评估脚本时，必须满足以下全部条件：

```
□ stride = 32 (不是 128)
□ 边缘裁剪 = 0 (不是 16)
□ 使用侵蚀标签 (gts_eroded / noBoundary)
□ Soft-logit 累积后 argmax (不是 per-patch argmax)
□ 批量推理 (batch_size ≥ 1, 建议 ≥ 14)
□ DSM per-tile min-max 归一化
□ RGB /255.0 归一化
□ 排除 clutter 类 (索引 5 或 255)
□ 仅报告前 5 类的 mIoU
□ Per-class OA 使用 Recall 公式 (TP/(TP+FN))
□ 混淆矩阵保存为原始 int64 计数
```

---

## 5. 已知结果

### 5.1 MFNet 论文 (Table I & II)

**Vaihingen:**
| 模型 | OA | mIoU | road | building | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Frozen (no adapter) | 88.01 | 75.11 | 89.51 | 94.64 | 71.71 | 89.47 | 76.83 |
| MMLoRA ViT-L | 91.31 | 83.20 | 93.27 | 97.21 | 78.24 | 91.02 | 87.18 |
| MMAdapter ViT-L | 92.02 | 83.69 | 92.59 | 96.29 | 80.15 | 93.09 | 89.08 |
| **MMAdapter ViT-H (best)** | **92.93** | **84.72** | 93.39 | 98.84 | 81.16 | 93.17 | 89.23 |

**Potsdam:**
| 模型 | OA | mIoU | road | building | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MMLoRA ViT-L | 90.99 | 85.71 | 92.68 | 97.59 | 88.34 | 88.57 | 96.35 |
| MMAdapter ViT-L | 91.62 | 86.37 | 93.69 | 98.31 | 87.27 | 88.78 | 96.29 |
| **MMAdapter ViT-H (best)** | **91.71** | **86.69** | 93.17 | 98.44 | 90.36 | 87.37 | 96.24 |

> per-class 列均为 Recall (= TP/(TP+FN))

### 5.2 我们的结果 (同协议)

参见 `/root/Mynet/mfnet_protocol_registry.py` — 这是权威数据源。

---

## 6. 与我们的 256² 协议的对比

我们日常使用的协议 (stride=128, trim=16, non-eroded labels) 与此协议有系统性差异：

| 参数 | 我们的 256² 协议 | MFNet 协议 | 影响 |
|------|:---:|:---:|:---|
| stride | 128 | **32** | MFNet 密集 ~16x 采样, 结果 ~+9pp |
| edge trim | 16 | **0** | MFNet 不裁剪边缘 |
| labels | 非侵蚀 | **侵蚀** | 侵蚀后边界像素被排除, 更难类别的边界像素不计入 |
| batch | 1 | **14** | MFNet 批量推理 |

**典型差异:**
```
同一模型:
  我们的 256² 协议   →  75-80% mIoU
  MFNet 协议          →  85-89% mIoU
  差距                →  ~9pp
```

**两个协议不可混用。** 对比 MFNet 论文必须用 MFNet 协议；内部模型间对比可以用我们的 256² 协议。
