# Plan4: SAM3 ViT 训练优化

> 日期：2026-05-07~08 (Phase 1) → 2026-05-08~ (Phase 2)
> 状态：**Phase 1 完成（负结果）→ Phase 2 规划中**
> 目标：突破 frozen bottleneck（73% → 78-83% → 85%）

---

## 反思：Plan1-4 失败路线的重新审视 (2026-05-08)

在继续 Plan4 之前，对之前所有"失败"路线做了根因分析。详细分析见对话记录，核心结论：

| 路线 | 原结论 | 反思结论 | 证据 |
|------|--------|---------|------|
| Plan1 P3: FPN Decoder | 微调比零样本差 | **实现不足，方向正确** | Plan3 用同范式达到 73.1%，证明 frozen+decoder 可行 |
| Plan1 P4: DSM Logit Bias | DSM 没用 | **方法错误（post-hoc乘法）** | MFNet 84.35% 证明 DSM 有用，但必须 learnable fusion |
| Plan4 Full SGD | 全量训练失败 | **执行不足，需要更多数据** | epoch 6 即过拟合，lr 太高，数据太少 |
| Plan4 D1/D2 Unfreeze | 解冻层数越多越差 | **粒度过粗** | 应该试 1/2/4 层而非直接跳到 8/16 |

**核心启示**：Plan1 P3 "fine-tune worse than zero-shot" 是整个项目最误导的结论。真相是那次的 FPN decoder (797K) + 多分类 loss + 不完整的梯度补丁失败了，但 frozen backbone + trainable decoder 范式是对的。**不要因为一次糟糕的实现否定一个方向。**

最大的未解决问题：**如何把 DSM 有效融入 SAM3？** Post-hoc multiplication 是错的，late SEFusion 提升有限（+1pp），in-ViT MMAdapter 是 MFNet 成功的关键但我们还没实现。

---

## Phase 1: 已完成实验 (2026-05-07~08)

### 动机

Plan3 最优 frozen 模型 (VPT+MFNet Decoder) = **73.54% mIoU** (256²)
MFNet SAM1 最佳 = **84.72%**。差距 11pp — 核心瓶颈在哪？

### D1: 解冻最后 8 层 ViT (AdamW, lr=1e-5)

```
SAM3 ViT: blocks 0-23 frozen, blocks 24-31 trainable
VPT Adapter: 2.1M | Decoder: 0.5M | ViT unfrozen: 113M
Total: 116M
```

| 指标 | 结果 |
|------|------|
| Crop best | 76.5% (epoch 2) |
| 256² mIoU | **72.48%** |
| vs Frozen | **-1.06pp** |
| 过拟合 | epoch 2 峰值后持续下降 |

### D2: 解冻最后 16 层 ViT (AdamW, lr=1e-5)

```
Total: 227M trainable
```

| 指标 | 结果 |
|------|------|
| Crop best | 76.1% (epoch 2) |
| 256² mIoU | **72.58%** |
| vs Frozen | **-0.96pp** |

### Full SGD: 456M 全量训练 (SGD+momentum, lr=0.01, MFNet 配方)

```
All ViT trainable + MFNet Decoder + DSM
SGD lr=0.01, momentum=0.9, weight_decay=5e-4
Warmup 2 epochs, CosineAnnealing 48 epochs
```

| 指标 | 结果 |
|------|------|
| Crop best | 76.3% (epoch 6) |
| 256² mIoU | 72.87% |
| 256² OA | 86.37% |
| 256² mF1 | 83.99% |
| vs Frozen | **-0.67pp** |

Per-class OA: road=88.5, building=91.9, grass=79.6, tree=82.8, car=73.8

## 全量训练 vs 冻结最优

| 指标 | Frozen (4.5M) | Full SGD (456M) | Δ |
|------|-------------|----------------|-----|
| OA | 86.44 | 86.37 | -0.07 |
| mIoU | **73.54** | 72.87 | **-0.67** |
| mF1 | **84.50** | 83.99 | -0.51 |
| Building OA | 90.5 | **91.9** | +1.4 |
| Grass OA | 78.3 | **79.6** | +1.3 |
| Tree OA | **84.7** | 82.8 | -1.9 |
| Car OA | **76.4** | 73.8 | -2.6 |

## Phase 2: 修正实验 (2026-05-08~)

Phase 1 的三个实验（D1/D2/Full SGD）都是在同一个假设下设计的：**"更多可训练参数 = 更高性能"**。这个假设被 Phase 1 的结果证伪了。Phase 2 需要换一个更细致的假设：**"正确的训练配方 × 正确的 DSM 融合 = 更高性能"**。

### 优先级排序

基于反思 + 最新论文（PointSAM/RefAtt-SAM/OVRS/SkySense++/CLFSam）的综合分析：

| 优先级 | 实验 | 预期收益 | 风险 | 时间 | 来源 |
|--------|------|---------|------|------|------|
| 🔴 P0 | D5: In-ViT DSM fusion | +5-8pp | 中 | 2-3天 | MFNet |
| 🔴 P0 | D8: MFNet MMAdapter 复现 | +5-8pp | 中 | 3-4天 | MFNet |
| 🟡 P1 | D3: LoveDA 预训练 | +3-5pp | 中 | 2-3天 | SkySense++ |
| 🟡 P1 | D6: Joint 训练 | +2-4pp | 低 | 1天 | — |
| 🟡 P1 | D9: HF-Adapter 边缘增强 | +1-3pp | 低 | 1天 | RefAtt-SAM, CLFSam |
| 🟡 P1 | D10: Rotation TTA | +1-2pp | 极低 | 半天 | OVRS |
| 🟢 P2 | D4: 细粒度 Unfreeze | +1-2pp | 低 | 1天 | — |
| 🟢 P2 | D7: Structure Loss | +1-2pp | 低 | 半天 | — |
| 🟢 P2 | D11: PBR 原型正则化 | +1-2pp | 低 | 1天 | PointSAM |
| 🟢 P2 | D12: RS-Token 域适配 | +0.5-1pp | 极低 | 半天 | RefAtt-SAM |

---

### D5: In-ViT DSM Fusion 🔴 最高优先级

**假设**: MFNet 的 85% vs 我们的 73% 差距中，最大的因子是 DSM 融合时机。MFNet 的 MMAdapter 在 ViT 每一层做 RGB↔DSM 交叉融合，我们的 SEFusion 是在 ViT 输出后才融合。

**当前实现 (Late Fusion)**:
```
RGB → SAM3 ViT (all 32 blocks) → vit_features
DSM → CNN Encoder → dsm_features
vit_features + dsm_features → SEFusion → Decoder
```
问题：DSM 从未参与 ViT 内部的 attention 计算。模型看不到"红色屋顶 + 高 DSM"这种联合信号。

**目标实现 (In-ViT Fusion)**:
```
RGB → [ViT Block 0] → [MMAdapter Fusion ← DSM_proj] → ... → [ViT Block 31] → Decoder
DSM → DSM Encoder → per-block projections → 注入每个 ViT block 的 Adapter
```

**实现方案**（仿 MFNet MMAdapter 但适配 SAM3 ViTDet）:

1. 在每个 ViTDet block 的 MLP 阶段之后，插入一个双分支 Adapter：
   ```python
   # RGB branch (VPT-style, already have this)
   rgb_adapted = relu(ln(rgb_feat) @ W_down) @ W_up
   # DSM branch (NEW)
   dsm_proj = dsm_proj_layers[block_idx](dsm_feat)  # align to RGB dim
   dsm_adapted = relu(ln(dsm_proj) @ W_down_shared) @ W_up_shared
   # Cross-modal fusion (NEW)
   out = orig_mlp(x) + λ1 * rgb_adapted + (1-λ1) * dsm_adapted
   ```
   关键：RGB 和 DSM 共享下投影/上投影权重（像 MFNet 那样），让两个模态在同一个低秩空间交互。

2. 不需要修改 SAM3 的 fused CUDA ops（Adapter 是纯 PyTorch 操作，在 ViT 主路径之外）。

3. 先用 frozen backbone 测试（只训练 Adapter + DSM projections），验证 in-ViT fusion 的提升幅度。若有效，再考虑配合全量/部分训练。

**成功标准**: 256² mIoU ≥ **78%**（比当前最佳 +5pp），证明 in-ViT fusion 是缩小与 MFNet 差距的关键。

**失败处理**: 如果 in-ViT fusion 提升 < 2pp，说明 SAM3 ViTDet 的结构（window attention）天然不适合跨模态融合——此时应考虑换 SAM1 ViT-H backbone。

---

### D3: LoveDA 预训练 → Vaihingen 微调

**假设**: Full training 在 Vaihingen 上失败是因为数据太少（960 patches for 456M params）。LoveDA 有 2522 张训练图（~40K+ patches），先在 LoveDA 上全量训练让 ViT 学会遥感特征，再在 Vaihingen 上微调（Adapter 或 low-lr partial）。

**实验设计**:
```
Stage 1 (LoveDA full train):
  - All ViT trainable, AdamW lr=1e-5, 10-20 epochs
  - 7-class output (LoveDA classes)
  - 目标：让 ViT 学会遥感域的基础特征

Stage 2 (Vaihingen fine-tune):
  - Replace decoder head (7→5 classes)
  - Option A: Freeze ViT + train Adapter (安全)
  - Option B: Full ViT with lr=1e-6 (激进)
  - 目标：迁移到 Vaihingen 5 类分割
```

**成功标准**: 256² mIoU ≥ **76%**（+3pp over frozen），证明预训练可以解决数据不足的问题。

---

### D6: Joint Vaihingen+Potsdam 训练

**假设**: 单一数据集样本量是瓶颈。联合训练提供 ~8000+ patches，可能让全量训练不再过拟合。

**实验设计**:
```
- 7 类输出 (Vaihingen 5 + Potsdam 不同类别的并集? 实际上两者都是 5 类)
- 实际是 5 类（两数据集类别一致）
- Full ViT trainable, AdamW lr=1e-5
- 每 batch 混合两个数据集的样本
- 分别在 Vaihingen 和 Potsdam 测试集上评估
```

**成功标准**: Vaihingen 256² mIoU ≥ **75%**，且 Potsdam 不退化。

---

### D4: Fine-grained Partial Unfreeze

**假设**: Phase 1 的 D1/D2 只试了 8 和 16 层，跨度太粗。最优解冻深度可能在 1-4 层。ViT 的浅层学习通用纹理，深层学习语义——遥感语义分割可能只需要调整最后几层。

**实验设计**:
```
D4a: Unfreeze last 1 layer  (~15M params)
D4b: Unfreeze last 2 layers (~30M params)
D4c: Unfreeze last 4 layers (~60M params)
D4d: Unfreeze last 1 layer + all LayerNorms (~17M params)
```

全部用 Adapter + Decoder + 解冻层，AdamW lr=1e-5，early stopping。

**成功标准**: 找到"解冻层数 → mIoU"的最优曲线。预期 2 层左右达到峰值 ≥ **74.5%**。

---

### D7: Structure Loss 替换 CE Loss

**假设**: Plan3 使用 edge-weighted BCE + IoU loss（`structure_loss`），而 Plan4 Full 用的是 CrossEntropyLoss。Structure loss 对边界精度更友好，可能缩小 Building/Road 的边界误差。

**实验设计**:
- 在 D5 (in-ViT fusion) 或 D3 (LoveDA pretrain) 的基础上
- 将 CE loss 替换为 `structure_loss` (edge-weighted BCE + weighted IoU)
- 对比两种 loss 的 per-class IoU，尤其关注边界类别（Building, Road）

**成功标准**: Building IoU 提升 +2pp 以上，mIoU 提升 +1pp。

---

### D8: MFNet MMAdapter 复现 🔴 最高优先级

**假设**: D5 (In-ViT DSM fusion) 的完整实现就是 MFNet 的 MMAdapter。这是 MFNet 论文中最核心的设计，也是我们与 85% 差距的最大来源。

**MFNet MMAdapter 工作原理**:
```
每个 ViT block 内部:
  MLP(x_rgb) → Adapter 分支 ─┐
                              ├→ λ₁*rgb_adapted + (1-λ₁)*dsm_adapted → output
  MLP(x_dsm) → Adapter 分支 ─┘

两个分支共享 W_down 和 W_up（在同一个低秩空间交互）
λ₁, λ₂ 是可学习的跨模态融合权重
```

与 D5 的区别：D5 是通用框架（"在 ViT 内部做融合"），D8 是 MFNet 论文的具体实现。两个互为补充。

**关键实现细节**（从 MFNet 论文）:
1. 仅替换 MLP 阶段的 Adapter 为 MMAdapter（attention 阶段保持单模态 Adapter）
2. DSM 投影到与 RGB 相同的维度后进入 ViT
3. 共享权重设计：W_down 和 W_up 两个模态共享，节省参数且强制对齐
4. λ₁, λ₂ 初始化为 0.5，可学习

**实现路径**:
```
当前 (VPT Adapter + SEFusion):
  ViT blocks (VPT) → Pyramid → SEFusion → Decoder

目标 (MMAdapter):
  ViT blocks (MMAdapter in each) → Pyramid → Decoder
  DSM → DSMEncoder → per-block projections → 注入每个 ViT block
```

**成功标准**: 256² mIoU ≥ **78%**（+5pp），证明 in-ViT fusion 是关键缺失要素。

**失败处理**: 如果 <75%（+2pp 以下），换 SAM1 ViT-H backbone（MFNet 已验证有效）。

---

### D9: HF-Adapter 边缘增强 🟡

**来源**: RefAtt-SAM 论文 — 在 Adapter 中注入图像的高频分量（边缘信息），改善遥感图像边界精度。

**原理**:
```
1. 输入图像 → FFT 高通滤波 → IFFT → 高频分量图
2. 高频分量通过一个小型 CNN (1-2层) → 高频特征
3. 高频特征与 ViT 中间特征 concat → 送入 Adapter
```

**为什么有效**: 遥感图像的 Building/Road 边界在 RGB 上经常模糊，但高频分量显式编码了边缘位置。当前 VPT Adapter 完全忽略了边缘信息。

**实现**:
```python
# 在 adapter_vit.py 的 AdapterBlock 中
def forward(self, x):
    # 原有 VPT adapter
    adapted = self.vpt_adapter(x)
    # 新增：高频边缘特征
    hf_feat = self.hf_proj(high_freq_map)  # 1×1 conv, align dims
    # 融合
    return adapted + 0.1 * hf_feat  # 小权重，避免干扰主特征
```

**成本**: ~10K 额外参数，推理开销极小（仅一个 1×1 conv）。

**成功标准**: Building IoU +1pp, Road IoU +1pp。

---

### D10: Rotation TTA（测试时旋转增强）🟡

**来源**: OVRS 论文 — 遥感图像无规范朝向，旋转 0/90/180/270 度后分别推理再融合，mIoU 提升 6pp（在 iSAID 上）。

**为什么有效**: 建筑、道路在遥感图像中朝向随机。SAM3 ViTDet 的 window attention 对朝向敏感，旋转后同一物体会得到不同特征。

**实现**:
```python
# 推理时
preds = []
for angle in [0, 90, 180, 270]:
    rotated_img = rotate(img, angle)
    logits = model(rotated_img)
    preds.append(rotate_back(logits.argmax(1), -angle))
final_pred = mode(preds)  # 投票
```

**成本**: 推理时间 ×4，但零训练成本。

**成功标准**: mIoU +1pp（无任何训练成本）。如果有效，可以进一步做训练时的 rotation augmentation。

---

### D11: PBR 原型正则化 🟢

**来源**: PointSAM 论文 — 用 FINCH 聚类对目标域特征做原型对齐，防止自训练漂移。可改造为训练正则化。

**原理**:
```
Offline: 用 frozen SAM3 提取 Vaihingen 训练集每个类别 gt 像素的特征 → FINCH 聚类 → 类原型
Online (训练时): 当前模型提取的特征 → 类原型 → cosine 相似度 → 原型对齐 loss
```

**为什么有效**: Vaihingen 只有 12 张图，模型容易忘记 SAM3 的预训练特征分布。原型正则化强制特征空间保持稳定，相当于"soft knowledge distillation"——原始 SAM3 是 teacher，当前模型是 student。

**实现**:
```python
# 训练 loss = task_loss + λ * prototype_loss
# prototype_loss = 1 - cos_sim(current_features, class_prototypes)
```

**成本**: 离线计算原型（~2分钟），训练时每 batch ~1ms。

**成功标准**: 训练更稳定（不出现过拟合），mIoU +1pp。

---

### D12: RS-Token 域适配 🟢

**来源**: RefAtt-SAM 论文 — 在 mask decoder 中加入可学习的域适配 token。

**改造为语义分割**:
```python
# 在 UNetFormerDecoder 的 bottleneck 处
self.rs_token = nn.Parameter(torch.zeros(1, 1, decode_channels))
# Forward: concat rs_token 到特征序列，过 attention 后再取出
```

**为什么有效**: SAM3 的预训练特征是为自然图像优化的。一个可学习的"遥感 token"在训练中学会遥感域的偏移量，通过 attention 传播到所有特征位置。

**成本**: 256 维 × 1 token = 256 个参数，几乎为零。

**成功标准**: mIoU +0.5pp。

---

## 论文参考速查

| 论文 | 年份/期刊 | 我们可利用的点 |
|------|----------|--------------|
| **MFNet** | 2025 IEEE TGRS | MMAdapter (in-ViT fusion) — 核心缺失 |
| **PointSAM** | 2025 IEEE TGRS | PBR 原型正则化（防过拟合） |
| **RefAtt-SAM** | 2026 IEEE TGRS | HF-Adapter（边缘）、RS-Token（域适配） |
| **OVRS** | IEEE TGRS | Rotation TTA、Scale-aware upsampling |
| **SkySense++** | 2025 Nature MI | 渐进式预训练策略（contrastive → semantic） |
| **CLFSam** | 2026 ISPRS J | Cooperative LoRA + 频率先验（待获取全文） |

---

## 路线图

```
Phase 2 (2026-05-08~15):
  Day 1-2 (零成本验证):
    D10: Rotation TTA → 验证 OVRS 的旋转增强是否有效
    D12: RS-Token → 验证域适配 token
    如果两者有效: mIoU 基线 +2pp (73→75%)

  Day 3-5 (低成本改进):
    D9: HF-Adapter → 边缘增强
    D11: PBR 原型正则化 → 防过拟合
    D7: Structure Loss → 边界精度
    目标: mIoU 76-77%

  Day 6-10 (核心攻坚):
    D5+D8: In-ViT DSM fusion (MMAdapter 复现)
    → 这是 Plan4 最核心的实验
    → 如果成功 (+5pp): mIoU 78-80%
    → 如果失败 (<2pp): 换 SAM1 ViT-H backbone

  Day 11-14 (规模化):
    D6: Joint Vaihingen+Potsdam training
    D3: LoveDA pretrain → Vaihingen fine-tune
    目标: mIoU 80-83%

Phase 3 (2026-05-16+):
  根据 Phase 2 结果：
  - 如果 D5/D8 有效：继续优化 in-ViT fusion → target 83-85%
  - 如果 D9 有效：探索 CLFSam 的 Cooperative LoRA + 频率先验
  - 如果 D3 有效：SkySense++ 式渐进预训练
  - 终极方案：SAM1 ViT-H + MMAdapter（MFNet 已证 85.03%）
```

---

## 关键设计原则（不变）

- 永远不修改 SAM3 的 fused CUDA operators（保持 frozen, no_grad）
- 可训练参数只注入在标准 PyTorch 层上
- 256² 滑动窗口为最终评估的唯一标准
- 每次评估必须输出四项指标：OA, mIoU, per-class IoU, per-class OA

---

## 评估输出标准

所有 evaluate/validate 函数必须输出四项指标：

| 指标 | JSON key | 说明 |
|------|----------|------|
| Overall OA | `avg_oa` | 全部前景类总体准确率 |
| Overall mIoU | `avg_miou` | 全部前景类平均 IoU |
| Per-class IoU | `per_class_iou` | 每个类别的 IoU |
| Per-class OA | `per_class_oa` | 每个类别的 OA: `(inter[c] + total - union[c]) / total` |
