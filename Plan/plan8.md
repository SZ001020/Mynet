# Plan8: 增强 In-ViT RGB↔DSM 交互

> 日期：2026-05-15
> 状态：**已完成 — 负面结果**
> 结论：Cross-attention (-0.26pp) 和 DSM attention bias (-0.65pp) 在 frozen ViT 中均无增益。Gate fusion 已足够。
> 来源：链路1 (Cross-Attention) 受 CMX/DFormer/RefAtt-SAM 启发；链路2 (Attention Bias) 受 TASAM/DFormer 启发

---

## 1. 核心判断

Plan6/7 的 in-ViT MMAdapter 中，RGB 和 DSM token 在注意力阶段彼此不可见，唯一的融合发生在 MLP 后的标量 gate。

Plan8 在**同一个 ViT block 的两个位置**增加跨模态交互：

```
  ┌── frozen self-attention ──────────────────────────────────┐
  │  Q @ K^T / sqrt(d)                                        │
  │       │                                                    │
  │       ├── ★ 干预点 B (链路2): DSM attn bias 加在 softmax 前  │
  │       ↓                                                    │
  │  softmax() @ V                                             │
  └──→ attn_x ────────────────────────────────────────────────┘
            │
            ├── ★ 干预点 A (链路1): cross-attn 加在 self-attn 之后
            ↓
       adapter → gate fusion → MLP → output
```

**两个干预点不冲突**，修改的是同一 forward pass 的不同阶段，共享对照组和所有基础设施。

---

## 2. 实验设计

### 2.1 训练协议

```text
起点: SAM3 base (frozen)，从头随机初始化所有可训练参数
数据集: Vaihingen
训练瓦片: MFNet train split (12 tiles)
验证瓦片: MFNet val split (4 tiles, 256² sliding window, stride=128, soft-logit)
Epochs: 8
Batch: 2
Crop: 512² online random
LR: 5e-5
Loss: structure_loss
分辨率: 1008
dsm_attn_mode: full
random_seed: 42 (所有实验组共用)
```

### 2.2 实验组总览

```
SAM3 base (frozen), seed=42
  │
  ├── CTRL: gate-only (同 Plan7-A 架构)
  │         验证管线，预期 ≈ 77%
  │
  ├── 链路 1 — Cross-Attention (4 全局 block)
  │   ├── CA-A: DSM→RGB 单向
  │   ├── CA-B: RGB↔DSM 双向
  │   └── CA-C: 双向 + per-token 门控
  │
  ├── 链路 2 — DSM Attention Bias (4 全局 block)
  │   ├── AB-A: 高程差 bias
  │   ├── AB-B: 可学习 DSM 相似度 bias
  │   ├── AB-C: Edge-aware bias (复用 prompt 特征)
  │   └── AB-D: Bias 结构消融 (factorized, per-head)
  │
  └── 链路 3 — 组合 (链路1/2 均有正向结果时触发)
      └── 最优 cross-attn + 最优 attn bias
```

### 2.3 执行顺序

```
Step 1: CTRL + CA-A + AB-A 三组并行 (3 张 GPU, ~3h)
        → 验证 CTRL 复现 Plan7-A
        → 分别判定 cross-attn 和 attn bias 方向是否有效

Step 2a: 若 CA-A 有效 → CA-B → CA-C
Step 2b: 若 AB-A 有效 → AB-B → AB-C → AB-D

Step 3: 若两条链路均有正向结果 → 链路3 组合
```

---

## 3. 链路 1: Cross-Attention RGB↔DSM

### 3.1 论文依据 (P8-1 输出)

| 论文 | 关键借鉴 | 设计决策 |
|------|---------|---------|
| **CMX** | FFM 只在 deep stages 做 cross-attn；双向 + Add&Norm + residual | 仅在全局 block 插入 |
| **DFormer** | Depth 只需影响 Q，不需影响 K/V；GAA 用 pooled-Q 降计算量 | 单向足以验证方向 |
| **RefAtt-SAM** | Cosine similarity > L1/L2；L2 norm 在 cross-attn 前很关键 | Pre-norm 在 Q/K/V 投影前 |

### 3.2 公共参数 (P8-1 推荐)

| 参数 | 值 | 依据 |
|------|-----|------|
| num_heads | 8 | 减半 SAM3 的 16 heads，降低注意力矩阵显存 |
| head_dim | 64 | 标准 ViT 惯例 |
| Q/K/V 投影 dim | 512 (8×64) | 减半从 1024，匹配 per-head dim |
| 输出投影 dim | 1024 | 恢复 token 维度用于残差连接 |
| Pre-norm | block.norm1 | 遵循 ViT pre-norm 惯例 |
| 插入 block | 仅 [7,15,23,31] | CMX 只在 deep stage 用；全局感受野才有意义的跨模态对应 |

### 3.3 插入位置 (P8-1 分析)

Cross-attn 在 self-attn 之后、adapter 残差之前：

```python
# Phase 1a: frozen self-attn (不变)
attn_x = self._attend(x)
attn_y = self._attend(y)

# Phase 1b: ★ cross-attn (新增)
xn = self.block.norm1(x); yn = self.block.norm1(y)
cross_y = CrossAttn(Q=yn, K=xn, V=xn)  # DSM→RGB
attn_y = attn_y + cross_y

# Phase 1c: adapter (不变)
x = x + drop_path(attn_x + rgb_attn_adapter(attn_x))
y = y + drop_path(attn_y + dsm_attn_adapter(attn_y))

# Phase 2: MLP + gate (完全不变)
```

**设计理由**: cross-attn 输出经过原有 adapter 再进入 gate，gate 收到的是已增强的特征；不新增 gate 项，降低训练不稳定性。

### 3.4 方案定义

#### CA-A: 单向 DSM→RGB (~2.1M/block, 4 block = ~8.4M)

```python
cross_y = CrossAttn(Q=norm(y), K=norm(x), V=norm(x))
y = y + drop_path(attn_y + dsm_ada(attn_y) + cross_y)
```

#### CA-B: 双向 RGB↔DSM (~4.2M/block, 4 block = ~16.8M)

```python
cross_y = CrossAttn(Q=norm(y), K=norm(x), V=norm(x))  # DSM→RGB
cross_x = CrossAttn(Q=norm(x), K=norm(y), V=norm(y))  # RGB→DSM
x = x + drop_path(attn_x + rgb_ada(attn_x) + cross_x)
y = y + drop_path(attn_y + dsm_ada(attn_y) + cross_y)
```

#### CA-C: 双向 + per-token 门控 (CA-B + ~4K/block)

```python
gate_x = Linear(cat([xn, yn]), 2048→1).sigmoid()
gate_y = Linear(cat([yn, xn]), 2048→1).sigmoid()
x = x + drop_path(attn_x + rgb_ada(attn_x) + gate_x * cross_x)
y = y + drop_path(attn_y + dsm_ada(attn_y) + gate_y * cross_y)
```

### 3.5 显存估算 (P8-1 分析)

| 方案 | 4 全局 block 参数 | 峰值激活增量 | 可行性 |
|------|------------------|-------------|--------|
| CA-A | 8.4M | ~1.0 GB | 安全 (余量 4GB) |
| CA-B | 16.8M | ~2.0 GB | 可行 |
| CA-C | 16.8M+16K | ~2.0 GB | 可行 (门控几乎免费) |

### 3.6 消融链路

```
CTRL ──→ CA-A: cross-attn 有无
CA-A ──→ CA-B: 单向 vs 双向
CA-B ──→ CA-C: 固定 vs 门控
```

---

## 4. 链路 2: DSM Attention Bias

### 4.1 机制

不新增 attention 模块，而是让 DSM 直接调制 frozen self-attention 的 Q@K^T 矩阵：

```python
# 当前
attn = softmax(Q @ K^T / sqrt(d)) @ V

# 链路 2
dsm_bias = BiasNet(dsm_features)       # DSM → [B, heads, N, N]
attn = softmax(Q @ K^T / sqrt(d) + dsm_bias * scale) @ V
```

### 4.2 与链路 1 的根本区别

| | 链路 1 (Cross-Attn) | 链路 2 (Attn Bias) |
|---|---|---|
| 机制 | DSM 作为 query 查询 RGB | DSM 改变 RGB 的 attention 分布 |
| 交互方向 | 跨模态 (DSM↔RGB) | 单模态内 (RGB 看 RGB 的方式被 DSM 调制) |
| 参数 | ~2M/block | ~0.05M/block |
| 显存 | ~1-2 GB 增量 | ~0.1-0.3 GB 增量 |

### 4.3 全局 block 的 O(N²) 问题

SAM3 全局 block 的 N = (1008/16)² = 3969，显式的 [B,heads,3969,3969] bias 矩阵 = ~2GB。

**解决方案: Factorized Bias**

```python
# 不用显式 N×N，用广播分解
bias_h = BiasNet_h(dsm_tokens)  # [B, heads, N, 1]  — 每行一个偏置
bias_w = BiasNet_w(dsm_tokens)  # [B, heads, 1, N]  — 每列一个偏置
dsm_bias = bias_h + bias_w      # [B, heads, N, N]  via broadcasting
```

物理含义：`bias_h[i]` 控制 token i "往外看"的总体倾向，`bias_w[j]` 控制 token j "被看到的"总体倾向。显存从 O(N²) → O(N)，3969² ≈ 15.7M → 2×3969 ≈ 8K。

### 4.4 方案定义

#### AB-A: 高程差 Bias (~0.05M/block)

```python
elev_i = dsm_tokens @ elev_proj  # [B, N, 1]
elev_diff = (elev_i - elev_i.T).abs()  # [B, N, N]
dsm_bias = BiasMLP(elev_diff)  # 1 → heads
dsm_bias = factorize(dsm_bias)  # N×N → N×1 + 1×N
```

最简验证。物理直觉：海拔相近 → 更可能同物 → 升高 attention。

#### AB-B: 可学习 DSM 相似度 Bias (~0.1M/block)

```python
dsm_q = BiasQ(norm(dsm_tokens))  # [B, N, head_dim]
dsm_k = BiasK(norm(dsm_tokens))  # [B, N, head_dim]
dsm_bias = dsm_q @ dsm_k.T       # [B, heads, N, N]
dsm_bias = factorize(dsm_bias)
```

模型自己学 DSM 空间中谁该关注谁，比 A 更有表达力。

#### AB-C: Edge-Aware Bias (~0.05M/block)

```python
# 复用 Plan7 已有的 prompt_tokens (Sobel edge + slope 特征)
dsm_bias = BiasFromPrompt(prompt_tokens)
dsm_bias = factorize(dsm_bias)
```

边界处降低 attention（跨边界信息不可靠），平坦处升高。

#### AB-D: Bias 结构消融

在最优 bias 源上做：
- **D1**: Factorized vs Full bias (窗口 block 可做 full)
- **D2**: Per-head vs Shared bias (不同 head 需要不同 bias 吗？)
- **D3**: Bias scale 消融 (0.1, 0.5, 1.0, 2.0)
- **D4**: 仅全局 vs 全 block (窗口 attention 中 bias 还有价值吗？)

### 4.5 消融链路

```
CTRL ──→ AB-A: attn bias 有无 (高程差)
AB-A ──→ AB-B: 几何 vs 可学习 bias
AB-B ──→ AB-C: DSM 特征 vs prompt 特征
AB-C ──→ AB-D: bias 结构精调
```

---

## 5. 停止条件与成功标准

### 5.1 链路 1

- CA-A < CTRL − 0.5pp → cross-attn 在 frozen ViT 中不适配，停止链路 1
- CA-B < CA-A − 0.5pp → 双向无增益，但 CA-C 仍可试（门控独立于方向数）
- CA-C < max(CA-A, CA-B) − 0.5pp → 门控无增益

### 5.2 链路 2

- AB-A < CTRL − 0.5pp → 高程差 bias 无效，但仍试 AB-B（可学习 bias 可能不同于几何 bias）
- AB-B < max(AB-A) − 0.5pp → 可学习 DSM 相似度无增益
- 后续按相同逻辑

### 5.3 成功标准

- > CTRL + 0.5pp = **有效改进**
- > CTRL + 1.0pp = **显著改进**，验证 Potsdam
- > Plan7-A + 1.0pp = **新 record**

---

## 6. 目录结构

```
RS-SAM3-p8/
├── ctrl_baseline/                  # 对照组
│   ├── model_ctrl.py
│   ├── train_ctrl.py
│   └── eval.py
├── chain1_cross_attn/
│   ├── cross_attn.py               # CrossAttentionAdapter
│   ├── mm_adapter_vit_ca.py        # MMAdapterPromptBlock + cross-attn 变体
│   ├── model_ca_a.py / train_ca_a.py / eval.py   # 单向
│   ├── model_ca_b.py / train_ca_b.py / eval.py   # 双向
│   └── model_ca_c.py / train_ca_c.py / eval.py   # 门控双向
├── chain2_attn_bias/
│   ├── attn_bias.py                # DSMAttentionBias + factorized bias
│   ├── mm_adapter_vit_ab.py        # MMAdapterPromptBlock + attn bias 变体
│   ├── model_ab_a.py / train_ab_a.py / eval.py   # 高程差
│   ├── model_ab_b.py / train_ab_b.py / eval.py   # 可学习相似度
│   └── model_ab_c.py / train_ab_c.py / eval.py   # edge-aware
├── chain3_combined/                # 组合实验 (条件触发)
└── shared/
    (复用 Plan7-A: dataset_adapter, dataset_online, mfnet_decoder, dsm_prompt, structure_loss)
```

---

## 7. Sub-Agent 任务拆分

### 任务 P8-1: Research-Planner — 论文分析 ✅ 已完成

P8-1 输出了 cross-attention 的完整参数推荐、插入策略、显存估算和共存策略。attn bias 链路已有足够设计细节，不需要额外的论文分析。

### 任务 P8-2: Build-Run — CTRL + CA-A + AB-A 并行实现

```
任务类型：代码生成 + 训练启动
背景事实：
  - P8-1 已完成，cross-attn 参数已确定 (8 heads, 64 dim, 512 proj, 4 global blocks)
  - CTRL = Plan7-A 架构(gate-only)，seed=42，从头训练
  - CA-A = CTRL + 单向 DSM→RGB cross-attn，seed=42
  - AB-A = CTRL + 高程差 factorized attn bias，seed=42
  - 三组并行，三张 GPU
当前目标：实现三组代码，smoke test，并行启动训练
输入文件：
  - /root/Mynet/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt/ (model.py, mm_adapter_vit.py, train_a.py, etc.)
  - /root/Mynet/plan8.md
  - P8-1 输出（参数推荐）
允许修改：RS-SAM3-p8/ 全部子目录
禁止修改：Plan7 代码、其他 plan 代码
输出要求：
  1. CTRL, CA-A, AB-A 三个目录的完整代码
  2. 三组 smoke test (前向+反向通过，维度检查)
  3. 三组训练命令 + tmux 窗口分配
  4. CTRL 初始 loss 作为基准 (CA-A, AB-A 应相近)
验收标准：
  - 三组无语法错误，训练正常启动
  - CTRL 的 adapter+prompt+decoder 参数需与 CA-A/AB-A 的相同部分完全一致
  - attn bias 使用 factorized 形式 (N×1 + 1×N)，不产生 O(N²) 显存爆炸
```

### 任务 P8-3: Eval-Analyze — Step 1 三组评估

```
任务类型：正式评估 + 消融分析
背景事实：CTRL, CA-A, AB-A 训练完成
当前目标：跑正式 256² sliding window 评估，三组互相对比
输入文件：三组 best_model.pt + Plan7-A eval JSON (外部参考)
输出要求：
  1. CTRL vs CA-A vs AB-A 三列对比表 (OA, mIoU, per-class IoU, per-class recall)
  2. CTRL vs Plan7-A 对比 (验证复现性)
  3. CA-A vs CTRL 差异分析 (cross-attn 独立贡献)
  4. AB-A vs CTRL 差异分析 (attn bias 独立贡献)
  5. 是否建议继续 CA-B 和/或 AB-B
验收标准：评估协议正确 (soft-logit + strict global)；CTRL ≈ Plan7-A ± 1pp
```

### 任务 P8-4: Build-Run — 链路后续阶段（依赖 P8-3）

```
任务类型：代码生成 + 训练启动 (条件执行)
背景事实：P8-3 已出结论
触发条件：
  - CA-A > CTRL + 0.3pp → 实现 CA-B
  - AB-A > CTRL + 0.3pp → 实现 AB-B
  - 后续按相同逻辑推进
输入文件：前阶段代码 + P8-3 评估结果 + plan8.md
输出要求：同 P8-2
```

### 任务 P8-5: Eval-Analyze — 后续阶段评估（依赖 P8-4）

```
任务类型：正式评估 + 消融链路分析
输入文件：后续阶段 best_model.pt
输出要求：
  1. 完整消融链路表 (每个阶段的变量、指标、结论)
  2. 链路 1 vs 链路 2 的横向对比 (cross-attn vs attn bias 哪个更有效)
  3. per-class 分析
  4. 是否触发链路 3 组合
```

### 任务 P8-6: Docs-Manage — 结果回写（依赖 P8-3, P8-5）

```
任务类型：实验记录 + 注册表更新
输入文件：所有阶段 eval JSON + plan8.md + model_registry.py
允许修改：plan8.md, model_registry.py
输出要求：
  1. plan8.md 写入所有阶段结论
  2. model_registry.py 写入 checkpoint (lineage: SAM3 base → CTRL/CA-*/AB-*)
  3. 消融链路完整记录：唯一变量、对照对象、结论
  4. 失败原因记录（如有）
```

---

## 8. 实验结果 (2026-05-16)

### 8.1 CTRL vs CA-A vs AB-A

256² sliding window, soft-logit accumulation, Vaihingen, 从头训练 (seed=42):

| Metric | CTRL | CA-A | ΔCA | AB-A | ΔAB |
|--------|------|------|-----|------|-----|
| **mIoU** | **74.77** | 74.52 | -0.26 | 74.12 | -0.65 |
| OA | 86.33 | 86.21 | -0.12 | 86.22 | -0.11 |
| road | 76.34 | 76.13 | -0.21 | 75.67 | -0.67 |
| building | 84.53 | 84.80 | +0.27 | 83.05 | -1.48 |
| grass | 62.47 | 61.47 | -1.00 | 63.13 | +0.66 |
| tree | 76.75 | 76.62 | -0.12 | 77.49 | +0.74 |
| car | 73.78 | 73.57 | -0.21 | 71.28 | -2.50 |

### 8.2 分析

- **CTRL (gate-only) 从零训练 = 74.77%**，比 Plan7-A (77.14%, 从 Plan6 checkpoint) 低 2.37pp。差距来自预训练 adapter 权重 vs 随机初始化。
- **CA-A (cross-attention) = -0.26pp**，噪声级别。DSM→RGB cross-attention 在 frozen ViT 中不提供额外信息——gate fusion 已让 DSM 信息充分进入 RGB 流。
- **AB-A (attention bias) = -0.65pp**，显著退化。因子化高程差 bias 干扰了 frozen attention 的原始分布。特别伤害 car (-2.50pp) 和 building (-1.48pp)，轻微改善 grass/tree——可能高程 bias 偏向植被特征。

### 8.3 结论

**两条链路均停止。** Cross-attention 和 attention bias 在 frozen SAM3 ViTDet + in-ViT MMAdapter 架构中均无增益。Gate fusion 在此规模下已足够有效。

此负面结果应写入 CLAUDE.md 的 "Failed directions" 部分，防止未来重复实验。

---

## 9. 代码实现关键约束

1. **CTRL 就是 Plan7-A 的复现**：从 SAM3 base 随机初始化，不加任何 cross-attn/bias，纯粹验证管线
2. **共享模块复用 Plan7-A**：dataset_adapter, dataset_online, mfnet_decoder, dsm_prompt, structure_loss
3. **gate 机制保留**：cross-attn 和 attn bias 都是额外路径，不删除原有 gate fusion
4. **frozen backbone 不动**：所有新增参数可训练，SAM3 ViT 完全冻结
5. **seed=42 全局锁死**：所有实验组共用，确保公平对比
6. **attn bias 必须 factorized**：全局 block 的 N=3969，显式 N×N bias 会导致 OOM
7. **显存监控**：三组并行时每张 GPU batch=2，1008²，需确认 CA-A 和 AB-A 不超 32GB
