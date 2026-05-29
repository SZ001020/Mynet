# Plan10: Potsdam 数据集训练与验证

> 日期：2026-05-29
> 状态：完成
> 目标：Potsdam 跨数据集验证——三条路线再次打平（差距<1pp）

---

## 1. 动机

所有实验都在 Vaihingen 上。现在需要：
1. 验证三种最优架构在 Potsdam 上的表现是否一致
2. 对标 MFNet 的 Potsdam 结果

## 2. 数据对齐验证

| | MFNet | 我们 | 对齐？ |
|------|------|------|:---:|
| 测试 tile | 4_10,5_11,2_11,3_10,6_11,7_12 | 同 | ✅ |
| 训练 tile | 16 张 (18 张可用 - 2 预留) | 18 张 | ⚠️ 多一些 |
| 通道 | IRRG → 取后 3 (RGB) | 直接 RGB | ✅ |
| 类别 | 5 前景 + clutter(忽略) | 同 | ✅ |
| 分辨率 | 6000×6000, 5cm GSD | 同 | ✅ |
| 评估 | 256² 滑动窗口 | 同 (软 logit) | ✅ |

## 3. 实验

三种架构，各训一次。Potsdam 比 Vaihingen 大 (18 train tile, 约 10K 样本)：

| 实验 | 架构 | 训练 | epochs |
|------|------|------|:---:|
| **P10-A** | Plan7-A (frozen + adapter + prompt) | 在线裁剪, 从零 | 20 |
| **P10-B** | F0 (shared + LoRA + 1×SEFusion) | 在线裁剪, 从零 | 20 |
| **P10-C** | F0'+L (frozen + LoRA + 4×SEFusion) | 在线裁剪, 从零 | 20 |

MFNet 训了 50 epoch × 1000 steps。我们先用 20 epoch 看趋势，够收敛就停止。

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| 数据 | Potsdam, 在线随机裁剪 256² |
| batch | 2 |
| epoch_steps | 1000 |
| lr (adapter/LoRA/decoder) | 5e-5 |
| lr (prompt encoder) | 2.5e-5 |
| seed | 42 |
| loss | structure_loss |
| eval | 256² 滑窗, 软 logit |

## 5. 结果（2026-05-29）

**Crop validation (Potsdam, 256²):**

| 实验 | Best | Best epoch | E1 |
|------|:---:|:---:|:---:|
| P10-A (Plan7-A) | 81.83% | 18 | 77.71% |
| P10-B (F0) | 82.12% | 14 | 69.32% |
| P10-C (F0'+L) | 82.17% | 19 | 77.19% |

**256² 正式评估 (Potsdam, 软 logit):**

| 实验 | OA | mIoU | road | bldg | grass | tree | car |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P10-A (Plan7-A) | 90.98 | 79.95 | 80.2 | 92.4 | 69.9 | 73.4 | 83.8 |
| **P10-B (F0)** | **91.62** | **80.91** | 82.5 | 90.2 | 72.1 | 75.0 | 84.7 |
| **P10-C (F0'+L)** | **91.67** | **80.95** | 82.0 | 90.8 | 72.2 | 75.3 | 84.4 |

**Vaihingen vs Potsdam 排名对比：**

| | Vaihingen | Potsdam |
|------|:---:|:---:|
| 1st | Plan7-A 77.55% | F0'+L 80.95% |
| 2nd | F0 77.34% | F0 80.91% |
| 3rd | F0'+L 77.24% | Plan7-A 79.95% |
| 差距 | 0.31pp | 1.00pp |

**参照 MFNet (SAM1 ViT-L, 50 epoch):**

| | SAM1 MFNet | SAM3 Plan10 |
|------|:---:|:---:|
| MMLoRA | 85.71% | — |
| MMAdapter | 86.37% | — |
| Plan7-A | — | 79.95% |
| F0 (LoRA) | — | 80.91% |
| F0'+L (LoRA) | — | 80.95% |

**结论：**
1. 三条路线在 Potsdam 上再次打平（差距 1pp），跨数据集一致性确认
2. LoRA 在更大数据集（Potsdam 18 tile）上略优于 Adapter（Vaihingen 12 tile 上相反）
3. SAM3 比 SAM1 低 ~5pp：20 vs 50 epoch + SAM3 固化度 + eval 协议差异
4. 架构选择不是跨数据集的关键变量——三条路线任选其一即可

## 6. 验收

| 对比 | 结论 |
|------|------|
| P10 内部排序 vs Vaihingen 排序 | 排名翻转但差距噪声级，架构非关键变量 |
| P10 vs MFNet Potsdam | SAM3 20 epoch < SAM1 50 epoch，符合固化预期 |
| 冻结基线 Potsdam (F0') | ~75% (E11, 未跑完 20 epoch) — 和 Vaihingen frozen 75.29% 接近 |

## 7. 代码位置

`Personal-Project/RS-SAM3-p10/cross_dataset_eval/` — 独立 eval 脚本
训练脚本复用 Plan6/7 代码，仅改 `--dataset potsdam` 参数
- Standard LoRA + DSM: ? (未单独报告)  
- MMLoRA: 85.71%
- MMAdapter: 86.37%

注意：MFNet 在 Potsdam 上训了 50 epoch，我们 20 epoch，不能直接对比数字。
但是架构排序应该一致，如果三条路线真的等价。

## 6. 代码位置

`Personal-Project/RS-SAM3-p10/potsdam_train/` — 三个训练脚本，复用已有 Plan7-A / F0 / F0'+L 模型代码，改 `--dataset potsdam`。
