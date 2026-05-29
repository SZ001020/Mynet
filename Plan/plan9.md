# Plan9: LoRA + DSM 结构先验融合

> 日期：2026-05-26
> 状态：Phase A 完成（无效），Phase B 规划中
> Phase A 结论：input-level edge/slope 拼接无法在 LoRA 基线上复现 Plan7-A 的 prompt 增益
> Phase B 方向：token-level PromptEncoder + gate 注入（与 Plan7-A 同范式，但用 LoRA 替代 Adapter）

---

## 1. 核心判断

Plan6/7 验证了三个独立有效因子：

| 因子 | 贡献 | 来源 |
|------|:---:|------|
| LoRA rank=8 (attn+MLP) | +1.95pp | Plan6 Phase4 F0'→F0'+L |
| DSM edge/slope prompt (token-level gate) | +0.57pp | Plan7-A |
| MFNetDecoder (GLA) | +0.7pp | Plan6 Phase3 |

但 LoRA 和 prompt **从未在同一注入层级上叠加过**：

- Plan7-A：frozen ViT + MMAdapter + PromptEncoder → prompt 通过 per-block gate 注入（**token 级别**）
- Phase A：LoRA ViT + edge/slope concat → prompt 拼在 DSM 通道上送入 encoder（**pixel 级别**）→ **无效**
- Phase B（新）：LoRA ViT + PromptEncoder + per-block gate → prompt 通过 per-block gate 注入（**token 级别**）

---

## 2. Phase A：Input-Level Edge/Slope 拼接 ❌

### 设计

```
F0'+L:
  DSM (1ch) → repeat(3) → SAM3 LoRA encoder → deepy

Phase A:
  DSM ─┐
  edge ─┼→ concat(3ch) → Conv1×1(3→3) → SAM3 LoRA encoder → deepy
  slope ─┘
```

不改模型架构、不碰 LoRA、不碰 decoder。只改 DSM 预处理——edge/slope 作为额外的输入通道。

### 实验矩阵

| 实验 | 初始化 | 改动 | 状态 |
|------|--------|------|:---:|
| **P9-A** | F0'+L (77.24%) | DSM+edge+slope → Conv1×1 → encoder | ❌ 完成 |
| P9-B | F0 (77.34%) | 同上 | 跳过 |

### 结果

```
                 crop val mIoU    正式 eval mIoU
F0'+L baseline:  76.81%           77.24%
P9-A run 1:      76.44%           —
P9-A run 2:      76.45%           —
Δ:               -0.36pp          —
```

两次重复实验一致（76.44% vs 76.45%），排除随机波动。

### 失败原因

1. **信息丢失在 patch_embed**。Edge/slope 作为原始像素通道送入 ViT 的 16×16 patch_embed，每个 patch 内的 Sobel/Laplacian 响应被均值池化——一个 16×16 区域（~1.4m²）内的平均边缘强度几乎不携带可用的结构信息。Plan7-A 的 PromptEncoder 用 4 级 stride-2 卷积保留了空间结构，然后投影到 1024-dim token 空间。

2. **LoRA 权重的输入分布被破坏**。F0'+L 的 LoRA 是在 "DSM repeat 3 次" 上训练的——三个通道完全相同。Phase A 换成 [DSM, slope, edge]，三个通道统计特性完全不同（高程值 vs 梯度幅值 vs 二阶导数），Conv1×1 虽可线性混合，但 LoRA 层的激活分布已经偏移。

3. **SAM3 预训练的通道语义不匹配**。SAM3 patch_embed 在自然 RGB 图像上预训练——三通道高度相关。DSM+edge+slope 之间没有这种相关性结构。

### 结论

**Input-level 通道拼接不能替代 token-level gate 注入。** Plan7-A 的 +0.57pp 收益来自 prompt 作为独立 token 流在每层参与 gate 融合，不是来自 edge/slope 信息本身。

---

## 3. Phase B：Token-Level PromptEncoder + Gate 注入（新）

### 核心思想

把 Plan7-A 的 prompt 注入机制搬到 LoRA 基线上：

```
Plan7-A:                              Phase B:
frozen ViT + MMAdapter                 LoRA ViT (from F0'+L)
  ↓                                     ↓
每层: frozen_SA(RGB)                   每层: LoRA_SA(RGB)
      frozen_SA(DSM)                         LoRA_SA(DSM)
      frozen_SA(prompt)                      frozen_SA(prompt)  ← 新增
      gate = softmax(w)                      gate = softmax(w)  ← 新增
      out = Σ gate_i * adapter_i(x)          out = gate[0]*RGB + gate[1]*DSM + gate[2]*prompt
                                               ← 无 adapter MLP, gate 直接加权 attn 输出
```

**关键差异**：
- Plan7-A 的 gate 加权的是 **adapter 输出**，attn 输出通过 adapter 变换后再 gate
- Phase B 的 gate 加权的是 **attn 输出本身**（LoRA 已经在 attn 中提供了适应能力，不需要额外的 adapter MLP）
- Prompt 流使用 frozen self-attention（不参与 LoRA），保证结构先验的稳定性

### 架构细节

```
输入:
  RGB → SAM3 patch_embed → RGB tokens [B, H/16, W/16, 1024]
  DSM → SAM3 patch_embed → DSM tokens [B, H/16, W/16, 1024]  
  DSM → dsm_edge_slope() → [slope, edge] → PromptEncoder → prompt tokens [B, H/16, W/16, 1024]

每个 ViT Block (32 blocks, global attn at [7,15,23,31]):
  1. Pre-norm: nx, ny, np = norm1(x), norm1(y), norm1(prompt)
  2. LoRA attention: ax, ay = lora_attn(nx, ny)    ← 共享 LoRA 权重
  3. Frozen attention: ap = frozen_attn(np)          ← 无 LoRA, 保持结构先验
  4. Gate: w = softmax(learnable_logits)             ← [3] 可学习标量
  5. Fuse: out = w[0]*ax + w[1]*ay + w[2]*ap        ← 直接加权 attn 输出
  6. Post-attn norm + MLP (LoRA MLP for x,y; frozen MLP for prompt)
  7. 最终输出: out_x = x + fuse_out_x, out_y = y + fuse_out_y
     (prompt 流不更新自身, 只参与 gate 注入)

After ViT:
  deepx, deepy = neck(x), neck(y)
  → FPN pyramids → 4×SEFusion at each scale → MFNetDecoder → logits
```

### 实验矩阵

| 实验 | 初始化 | 改动 | epochs |
|------|--------|------|:---:|
| **P9-B2** | F0'+L best (77.24%) | 新增 PromptEncoder + per-block gate | 8 |

只做一个实验。P9-B（从 F0 出发）跳过——先验证 LoRA + prompt gate 是否能叠加，成功了再扩大。

### 预期

```
如果 P9-B2 > 77.55%: LoRA + prompt gate > Plan7-A, 两个有效因子成功叠加
如果 P9-B2 ≈ 77.24%: prompt gate 在 LoRA 下无额外增益（LoRA 已覆盖 prompt 的贡献）
如果 P9-B2 < 77.24%: prompt gate 干扰 LoRA（类似 Phase4 F1 中 LoRA + adapter 互斥）
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 数据 | Vaihingen, 在线随机裁剪, 512² |
| batch | 2 |
| epoch_steps | 1000 |
| epochs | 8 |
| lr | 5e-5 |
| seed | 42 |
| loss | structure_loss |
| eval | 256² 滑窗, 软 logit 积累 |
| init_from | F0'+L best (plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt) |

### 实现要点

- **PromptEncoder**：复用 Plan7-A 的 `PromptEncoder`（`dsm_prompt.py`）——4 级 stride-2 卷积 stem + 1×1 投影到 1024-dim
- **Block 改造**：在 F0'+L 的 LoRA block 基础上，增加 prompt token 的 frozen self-attention 通路 + 3-way gate
- **Gate 初始化**：`logits = [1.0, 1.0, 0.1]`——初始时 prompt 权重很小，让 LoRA 流主导，逐步学习 prompt 的贡献
- **LoRA 权重**：从 F0'+L checkpoint 加载，保持冻结（不参与 LoRA 训练的部分仍然是 frozen）
- **新增可训练参数**：PromptEncoder (~0.3M) + per-block gate logits (32×3=96 个标量) + per-block prompt attn QKV 投影

### 代码位置

`Personal-Project/RS-SAM3-p9/phase_b_token_gate/`

---

## 4. 验收

| 结果 | 动作 |
|------|------|
| P9-B2 < F0'+L (77.24%) | prompt gate 干扰 LoRA，Plan9 彻底关闭 |
| P9-B2 = 77.24-77.55% | prompt 在 LoRA 下弱有效，不扩大 |
| P9-B2 > 77.55% | LoRA+prompt gate 超越 Plan7-A，当前最优 |
| P9-B2 > 78% | LoRA+prompt 互补性确认，可扩大实验矩阵 |

---

## 5. Sub-agent 分派

| Agent | 任务 |
|-------|------|
| Build-Run | 基于 F0'+L 代码实现 Phase B block 改造 + PromptEncoder 集成 + smoke test |
| Build-Run | 启动 P9-B2 训练 (8 epoch) |
| Eval-Analyze | 256² 正式 eval + 与 F0'+L / Plan7-A 对比分析 |
| Docs-Manage | 更新 plan9.md 结果 + model_registry.py + CLAUDE.md |
