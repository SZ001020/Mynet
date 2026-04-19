# 第六周计划：高低植被判别修复（DSM 驱动）

## 1. 周目标
- 解决 high vegetation 与 low vegetation 的系统性混淆。
- 在 same-domain 协议下先完成机制闭环，再决定是否回灌到 weak-cross-domain。
- 形成可解释证据，不只追求单次 mIoU 峰值。

## 2. 问题诊断
- RGB/NIR 外观相近时，树冠与草地容易互相吞并。
- 跨域场景下颜色分布漂移，固定阈值伪标签不稳。
- 当前 DSM 更像是附加输入，缺少显式高度先验约束。

## 3. 本周技术主线（低侵入优先）

### 3.1 P0：DSM 先验显式化（必须完成）
1. 高度一致性损失（首优）

$$
L_{hc}=\lVert p_{tree}-g(h)\rVert_1 + \lVert p_{lowveg}-(1-g(h))\rVert_1
$$

- 其中 $g(h)=\sigma((h-\mu_h)/\tau_h)$，$h$ 为归一化 DSM。
- 目标：tree 概率与高程正相关，lowveg 与低高程正相关。
- 改动位置：`MFNet/train_uda_struct_v1.py`。

2. 类别感知伪标签阈值
- tree 与 lowveg 使用不同阈值，不再共享 fixed threshold。
- 叠加 DSM 门控：中低高度区抑制 tree 伪标签，高高度区抑制 lowveg 伪标签。
- 改动位置：`MFNet/train_uda_struct_v1.py`、`MFNet/utils.py`。

### 3.2 P1：DSM 特征增强（建议完成）
1. DSM 导数特征
- 在 DSM 分支加入 slope、local roughness、local relative height（减局部均值）。
- 改动位置：`MFNet/utils.py`（数据预处理）。

2. 统一 DSM 归一化
- 用训练集统计量标准化（或分域统计），替代逐图 min-max。
- 改动位置：`MFNet/utils.py`。

### 3.3 P2：结构增强（条件开启）
1. 植被辅助头
- 先做 vegetation vs non-vegetation，再在 vegetation 内细分 high/low。
- 仅在 P0/P1 稳定后启用，防止本周任务爆炸。
- 改动位置：`MFNet/UNetFormer_MMSAM.py`、`MFNet/train_uda_struct_v1.py`。

## 4. 一周执行排期

### Day1：基线复核与统计准备
- 固定 same-domain 配置与 seed 列表（至少 3 个）。
- 统计 DSM 全局/分域均值方差，导出配置快照。
- 输出：`week6_dsm_stats.json`、baseline 混淆矩阵（含 tree/lowveg）。

### Day2：接入 P0-1 高度一致性损失
- 在训练损失中加入 $L_{hc}$，支持权重开关与日志记录。
- 小规模 sanity run（1 seed）验证收敛与数值稳定。
- 输出：loss 曲线、首版对比表（baseline vs +L_hc）。

### Day3：接入 P0-2 类别感知阈值 + DSM 门控
- 实现 tree/lowveg 双阈值与高度门控伪标签。
- 与 Day2 做 1 seed 对比，检查伪标签覆盖率与类别占比。
- 输出：伪标签统计图、困难样本可视化。

### Day4：接入 P1 DSM 导数特征
- 增加 slope/roughness/relative height 特征并验证输入分布。
- 进行 1 seed 对比，观察 tree/lowveg IoU 与混淆变化。
- 输出：特征统计与 ablation 初表。

### Day5：多 seed 稳定性复验（核心日）
- 对 baseline、+L_hc、+L_hc+class-thr、+L_hc+class-thr+DSM-feat 各跑 3 seed。
- 汇总 mean±std，重点看 tree/lowveg 的 IoU 与互相误分率。
- 输出：T3/T4 初版、F4 稳定性图。

### Day6：条件开启 P2（可选）
- 仅当 Day5 显示稳定正增益时，接入植被辅助头并做 1 到 2 seed 验证。
- 若未通过门槛，P2 直接降级为备选，不进入主线。
- 输出：P2 去留结论。

### Day7：封板与回灌决策
- 固化本周通过门槛的配置白名单。
- 产出回灌建议（是否进入 week7 weak-cross-domain）。
- 输出：week6 总结报告、回灌清单。

## 5. 实验矩阵（same-domain）
- A0：Baseline。
- A1：A0 + $L_{hc}$。
- A2：A1 + 类别感知阈值 + DSM 门控。
- A3：A2 + DSM 导数特征。
- A4（可选）：A3 + 植被辅助头。

## 6. 记录与分析指标
- 主指标：mIoU、mF1、OA。
- 类别指标：IoU_tree、IoU_lowveg、F1_tree、F1_lowveg。
- 结构指标：tree→lowveg 与 lowveg→tree 混淆比例。
- 稳定性指标：3 seed 的 mean±std。
- 伪标签指标：tree/lowveg 覆盖率、置信度分布、门控前后占比。

## 7. 验收门槛（本周）
- A2 或 A3 相比 A0，在 same-domain 上满足：
	- IoU_tree 与 IoU_lowveg 至少一个显著提升，另一个不恶化超过 0.3。
	- 两者互相误分率下降。
	- 多 seed 下方向一致，std 不显著放大。
- 未过门槛的模块不进入第七周跨域回灌。

## 8. 风险与回退
- 风险：新增损失导致训练震荡。
	- 回退：先降低 $\lambda_{hc}$，必要时仅保留类别阈值与门控。
- 风险：DSM 特征通道引入噪声。
	- 回退：仅保留 relative height，去掉 roughness。
- 风险：辅助头带来额外不稳定。
	- 回退：本周不纳入主线，仅保留分析结果。

## 9. 交付件清单（Week6）
- T3-final：组件消融表（含 A0 到 A4）。
- T4-v1：模态与微调策略初表（补高低植被专门列）。
- F3-v3：高低植被困难区域边界对比图。
- F4-v4：多 seed 稳定性曲线与误分趋势图。
- `week6_summary.md`：结论与 Week7 回灌白名单。
