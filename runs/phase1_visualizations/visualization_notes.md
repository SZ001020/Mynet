# Phase 1 推理图对比分析

> 生成时间: 2026-04-29T17:54:18.904498
> 输出目录: /root/Mynet/autodl-tmp/runs/phase1_visualizations

## 对比配置

| 缩写 | 含义 |
|------|------|
| Instance | Transformer Decoder 仅实例 mask |
| Instance+Pres | + Presence Score 过滤 |
| Semantic | Segmentation Head 仅语义 mask |
| Semantic+Pres | + Presence Score 过滤 |
| Dual-Head | 双头融合 (element-wise max) |
| Dual-Head+Pres | 双头 + Presence (默认配置) |

## 每个样本的输出文件

- `comparison_{dataset}_{tile}.png` — 8 图对比 (RGB + GT + 6 配置)
- `{dataset}_{tile}_{config}.png` — 单张配置预测图

## 观察要点

1. **Semantic vs Instance**: 注意密集区域（道路/草地）和离散物体（建筑/车辆）在两个 head 下的差异
2. **Presence 效果**: 对比 ±Presence 配置，观察误检的抑制情况
3. **Dual-Head 融合**: 观察双头融合是否能同时保留实例边界和语义覆盖
4. **Clutter 问题**: 注意 clutter（红色）类的分布——这是 SAM 3 最弱的类别
