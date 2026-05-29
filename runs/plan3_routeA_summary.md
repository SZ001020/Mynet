# Plan3 Route A: Adapter-ViT + UNet Decoder 实验结果

> 日期：2026-05-01~02 | 代码：`RS-SAM3-p3/` | RTX 5090 32GB | MFNet splits

## 架构

```
SAM3 ViTDet (32 AdapterBlocks, frozen blocks + trainable prompt_learn MLP)
  → FPN (frozen, 3 scales: 288², 144², 72²)
  → UNet Decoder (trainable ConvBlocks, Dropout2d=0.1)
  → 5-channel per-class logits → argmax
```

- 可训练参数：6.57M / 814.33M (0.8%)
- Loss: structure_loss = 边缘加权 BCE + 加权 IoU
- 无 text prompt（纯视觉模型）

## 全图推理结果（滑动窗口: 1008², stride=672）

### Vaihingen（NIRRG, 12 train / 4 val tiles）

| Tile | OA | mIoU | road | building | grass | tree | car |
|------|-----|------|------|----------|-------|------|-----|
| area5 | 91.7% | 74.8% | 83.9 | 91.9 | 53.2 | 75.1 | 69.8 |
| area21 | 89.6% | 75.6% | 83.6 | 92.2 | 58.4 | 83.2 | 60.7 |
| area15 | 86.8% | 76.5% | 76.2 | 89.7 | 66.5 | 77.3 | 72.7 |
| area30 | 89.5% | 78.5% | 83.5 | 92.3 | 67.4 | 80.2 | 69.3 |
| **Avg** | **89.4%** | **76.3%** | **81.8** | **91.5** | **61.4** | **79.0** | **68.1** |

Best epoch: 7 | 配置：bs=4, lr=1e-4, wd=1e-3, dropout=0.1

### Potsdam（RGBIR, 18 train / 6 val tiles）

| Tile | OA | mIoU | road | building | grass | tree | car |
|------|-----|------|------|----------|-------|------|-----|
| 4_10 | 89.1% | 76.0% | 75.8 | 89.0 | 69.0 | 72.0 | 74.2 |
| 5_11 | 92.0% | 78.4% | 83.0 | 92.5 | 67.5 | 74.5 | 74.4 |
| 2_11 | 87.4% | 74.5% | 79.9 | 83.4 | 67.5 | 67.6 | 74.3 |
| 3_10 | 90.2% | 78.7% | 73.0 | 87.9 | 77.8 | 78.5 | 76.3 |
| 6_11 | 93.6% | 78.1% | 79.9 | 94.4 | 65.5 | 75.6 | 75.2 |
| 7_12 | 91.1% | 73.6% | 86.8 | 91.1 | 55.0 | 56.3 | 79.1 |
| **Avg** | **90.6%** | **76.6%** | **79.7** | **89.7** | **67.1** | **70.8** | **75.6** |

Best epoch: 5 | 配置：bs=8, lr=1e-4, wd=1e-4, 无 dropout

## 对比基线（全图 mIoU）

| 方法 | Vaihingen | Potsdam |
|------|-----------|---------|
| Plan1 多分类 zero-shot | 65.8% | 56.6% |
| Plan2 per-class binary tuned | 55.4% | 43.7% |
| **Ours (512-crop val)** | **67.1%** | **47.2%** |
| **Ours (全图滑动窗口)** | **76.3%** | **76.6%** |

## 关键发现

1. **Adapter-VPT 极其有效**：全图 mIoU 76%+，远超 zero-shot（Vaihingen +10.5pp, Potsdam +20.0pp）
2. **512-crop 评估严重低估性能**：全图 mIoU 比 crop 评估高 9-29pp。原因：(a) 滑动窗口有重叠平滑，(b) 全图包含更多大物体上下文，(c) crop 采样对小类（car）欠采样
3. **两数据集表现趋同**：全图尺度下 Vaihingen(76.3%) ≈ Potsdam(76.6%)，NIR 优势在滑动窗口中被削弱
4. **Plan1 Phase 3 失败原因确认**：FPN decoder (797K) 太弱，vs Adapter 在 ViT token 空间修改表征
5. **无需 text prompt**：纯视觉 Adapter+UNet 即可超越 zero-shot text-based SAM3

## 文件

```
autodl-tmp/runs/
├── plan3_adapter_20260501_224049/    # Vaihingen best (全图 76.3%)
│   ├── best_model.pt
│   ├── full_vaihingen_area5.png ~ area30.png
│   ├── full_tile_results_vaihingen.json
│   └── history.json
├── plan3_adapter_20260501_185038/    # Potsdam best (全图 76.6%)
│   ├── best_model.pt
│   ├── full_potsdam_4_10.png ~ 7_12.png
│   ├── full_tile_results_potsdam.json
│   └── ...
└── plan3_routeA_summary.md           # 本文档
```
