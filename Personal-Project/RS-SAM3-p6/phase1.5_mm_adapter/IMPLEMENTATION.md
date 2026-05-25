# Plan6 Phase 1.5: 数据对标 + LoRA 协同

> 交接给 codex 实现。Phase 1 的 MMAdapter（76.86% mIoU）已完成，在此基础上做两项改进。
> 所有新代码放在当前目录 `RS-SAM3-p6/phase1.5_mm_adapter/`，从 Phase 1 复制文件开始修改。

---

## 背景

Phase 1 的两个已知问题：

1. **数据使用效率低**：预提取 2,800 个固定 256² 窗口，每 epoch 看到相同数据。MFNet 每 epoch 在线随机裁剪 10,000 个不重复位置，50 epoch 总样本 500,000 vs 我们 56,000，差 9 倍多样性。

2. **只有 MMAdapter，没有 LoRA**：MFNet Best（84.72%）的核心是 LoRA + MMAdapter 协同工作——LoRA 修正 RGB attention，MMAdapter 注入 DSM 做跨模态融合。Phase 1 只有后者。

Phase 1.5 的目标：在不改变整体架构的前提下，数据量对齐 MFNet + 加入 LoRA，预期 mIoU 从 76.86% 提升到 78-80%。

---

## 第一步：在线随机裁剪数据集

### 当前问题

`train_phase1.py` 中的 `Window256DatasetDSM` 在 `__init__` 中预提取所有窗口到 `self.samples` 列表：

```python
for tile in tiles:
    for y in range(0, h - 128, stride):
        for x in range(0, w - 128, stride):
            self.samples.append((patch, dsm_patch, label))
```

每个 epoch 遍历同一个 `self.samples` 列表，模型反复看到完全相同的 2,800 个裁剪。

### 目标实现

新建 `dataset_online.py`，实现 `OnlineCropDataset`。核心思路来自 MFNet `utils.py:ISPRS_dataset`：

1. `__init__` 中把全部 tile 的 RGB、DSM、label 加载到内存（MFNet 的 CACHE=True 策略），存为三个列表
2. `__len__` 返回 `BATCH_SIZE * epoch_steps`（epoch_steps 可配置，默认 1000，对标 MFNet）
3. `__getitem__` 每次：
   - 从 tile 列表中随机选一张
   - 从该 tile 中随机 crop 一个 256×256 区域
   - 应用 flip/rotate augmentation（与 Phase 1 一致）
   - 返回 (RGB tensor, DSM tensor, label tensor)

```python
# dataset_online.py 结构示意

class OnlineCropDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix,
                 dsm_paths, crop_size=256, epoch_steps=1000, batch_size=None):
        self.crop_size = crop_size
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size or 4

        # 将所有 tile 加载到内存（对标 MFNet CACHE）
        self.images = []     # list of (H, W, 3) uint8 numpy
        self.dsms = []       # list of (H, W) float32 numpy
        self.labels = []     # list of (H, W) int64 numpy (class index)

        for tile in tiles:
            # 加载 RGB、转换 label、加载并归一化 DSM
            ...

    def __len__(self):
        return self.batch_size * self.epoch_steps

    def __getitem__(self, idx):
        tile_idx = random.randint(0, len(self.images) - 1)
        img = self.images[tile_idx]
        dsm = self.dsms[tile_idx]
        label = self.labels[tile_idx]

        H, W = img.shape[:2]
        y = random.randint(0, max(0, H - self.crop_size))
        x = random.randint(0, max(0, W - self.crop_size))

        img_crop = img[y:y+self.crop_size, x:x+self.crop_size]
        dsm_crop = dsm[y:y+self.crop_size, x:x+self.crop_size]
        label_crop = label[y:y+self.crop_size, x:x+self.crop_size]

        # Augmentation: flip (50%), mirror (50%), rotate 0/90/180/270
        ...

        return (torch.from_numpy(img_crop).permute(2,0,1).float()/255.0,
                torch.from_numpy(dsm_crop).float(),
                torch.from_numpy(label_crop).long())
```

### 关键细节

- **DSM 归一化**：每张 tile 的 DSM 独立做 min-max normalize 到 [0, 1]。在 `__init__` 加载时就做好，不要每次 crop 重复计算
- **边界处理**：如果 tile 尺寸小于 crop_size（Vaihingen tile 约 1900×2500，256² 不存在此问题），padding 到 crop_size
- **Ignore 像素**：label 中 IGNORE_INDEX (255) 的像素保留原值，不做特殊处理
- **augmentation 一致性**：flip/mirror/rotate 必须同时应用到 RGB、DSM、label 三个数组
- **epoch_steps 默认值**：Vaihingen 用 1000（MFNet 默认），Potsdam 用 1000
- **数据路径**：沿用 Phase 1 `dataset_paths()` 函数中定义的路径

### 训练脚本修改

复制 `train_phase1.py` → `train_phase1_5.py`，只改三处：

1. import 新 dataset：`from dataset_online import OnlineCropDataset`
2. DataLoader 中 `drop_last` 去掉（因为 dataset 长度已经对齐 batch_size）
3. 打印信息中显示 epoch_steps 和实际样本量

其余（模型加载、优化器、训练循环、验证）完全不变。

---

## 第二步：LoRA + MMAdapter 协同

### 当前架构回顾

Phase 1 的 `MMAdapterBlock.forward` 流程：

```
输入: RGB tokens (x), DSM tokens (y)
1. Frozen block attention on x (window attention)
2. Adapter residual: attn_x + rgb_attn_adapter(attn_x)
3. Adapter on y: dsm_attn_adapter(norm(y))  或 frozen attn on y
4. Frozen MLP on x
5. Cross-modal MLP fusion:
   x = x + mlp_x + wx * rgb_mlp_adapter(xn) + (1-wx) * dsm_mlp_adapter(yn)
   y = y + mlp_y + wy * dsm_mlp_adapter(yn) + (1-wy) * rgb_mlp_adapter(xn)
```

所有 `block.attn` 和 `block.mlp` 的权重是 frozen 的。目前 adapter 是唯一可训练的部分。

### LoRA 注入策略

LoRA 注入到每个 block 的 **frozen attention 线性层**和 **frozen MLP 线性层**，与 MMAdapter 的 adapter 侧同时工作：

```
Block 内部（展开 3 层 LoRA）:
  attn.qkv:  LoRA(Linear(1024, 3072), rank=8)  → 修正 QKV 投影
  attn.proj: LoRA(Linear(1024, 1024), rank=8)  → 修正 attention 输出
  mlp.fc1:   LoRA(Linear(1024, 4096), rank=8)  → 修正 MLP 第一层
  mlp.fc2:   LoRA(Linear(4096, 1024), rank=8)  → 修正 MLP 第二层
```

**只在 RGB stream 的 block 里注入 LoRA**。DSM stream 的 attention 走的是 `dsm_attn_adapter`（轻量模式），不需要 LoRA。

### 为什么这次 LoRA 会有效

Phase 3 的 LoRA 失败的原因：
1. 无 DSM → LoRA 只能学 RGB 内部修正，收益有限
2. DSM 在 ViT 外部融合 → LoRA 看不到高程信息

Phase 1.5 的 LoRA + MMAdapter 不同：
1. MMAdapter 已经把 DSM token 注入 ViT 每一层
2. LoRA 修正 QKV 时，attention 计算本身就包含了 RGB×DSM 的交互
3. LoRA 修正 MLP 时，cross-modal gate fusion（wx/wy）已经混合了两个模态
4. LoRA 可以学到"当 DSM 表示高处时，Q 应该更关注那些 token"

### 实现方案

修改 `mm_adapter_vit.py`，新增 `inject_lora_into_mm_block` 函数：

```python
# mm_adapter_vit.py 新增内容

class LoRALinear(nn.Module):
    """在 frozen nn.Linear 上注入 LoRA 低秩修正。

    forward: y = Wx + (alpha/rank) * B(A(x))
    A, B 可训练；W 冻结。
    """
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        for p in self.original.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return result + lora_out


def inject_lora_into_mm_block(mm_block: MMAdapterBlock, rank: int = 8, alpha: float = 16.0):
    """在 MMAdapterBlock 的 frozen block 内部注入 LoRA。

    注入目标（仅 RGB stream 的 block）：
      block.attn.qkv   — fused QKV projection
      block.attn.proj  — attention output projection
      block.mlp.fc1    — MLP first layer
      block.mlp.fc2    — MLP second layer

    LoRA 参数命名规则：
      block.attn.qkv → lora_qkv_A, lora_qkv_B, ...
      用于后续 requires_grad 分组。
    """
    inner = mm_block.block  # 原始 frozen SAM3 ViT block

    # QKV (fused linear)
    if hasattr(inner.attn, 'qkv'):
        inner.attn.qkv = LoRALinear(inner.attn.qkv, rank=rank, alpha=alpha)

    # Attention output projection
    if hasattr(inner.attn, 'proj'):
        inner.attn.proj = LoRALinear(inner.attn.proj, rank=rank, alpha=alpha)

    # MLP layers
    if hasattr(inner.mlp, 'fc1'):
        inner.mlp.fc1 = LoRALinear(inner.mlp.fc1, rank=rank, alpha=alpha)
    if hasattr(inner.mlp, 'fc2'):
        inner.mlp.fc2 = LoRALinear(inner.mlp.fc2, rank=rank, alpha=alpha)
```

### 参数分组与优化器配置

LoRA 和 MMAdapter 需要不同学习率（参考 MFNet 对环境变量 `SSRS_BASE_LR` 的使用，MFNet 全局 SGD lr=0.01）：

```python
# train_phase1_5.py 优化器部分

lora_params = []
adapter_params = []  # MMAdapter + DSM encoder + decoder
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'lora_' in name:
        lora_params.append(p)
    else:
        adapter_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': adapter_params, 'lr': args.lr},       # 1e-4 (MMAdapter)
    {'params': lora_params, 'lr': args.lora_lr},     # 5e-5 (LoRA, 更保守)
], weight_decay=1e-3)
```

### 新增命令行参数

在 `train_phase1_5.py` 中新增：

```python
parser.add_argument('--lora-rank', type=int, default=8)
parser.add_argument('--lora-alpha', type=float, default=16.0)
parser.add_argument('--lora-lr', type=float, default=5e-5)
parser.add_argument('--epoch-steps', type=int, default=1000)
```

---

## 预期结果

| 方案 | mIoU | 说明 |
|------|------|------|
| Phase 1 baseline | 76.86% | MMAdapter only, fixed 2,800 windows |
| + 在线裁剪 | ~77.5% | 数据多样性 9× 提升 |
| + 在线裁剪 + LoRA | ~78-80% | LoRA 与 MMAdapter 协同 |

---

## 需要创建/修改的文件

```
RS-SAM3-p6/phase1.5_mm_adapter/    ← 新建目录
├── dataset_online.py               ← 新建（在线随机裁剪数据集）
├── mm_adapter_vit.py               ← 从 Phase 1 复制，新增 LoRALinear + inject_lora_into_mm_block
├── mfnet_decoder.py                ← 从 Phase 1 复制（不变）
├── structure_loss.py               ← 从 Phase 1 复制（不变）
├── model_phase1.py                 ← 从 Phase 1 复制，新增 lora_rank/lora_alpha 参数
├── train_phase1_5.py               ← 从 Phase 1 复制并修改（新 dataset + LoRA 配置）
└── eval_mfnet_protocol.py          ← 从 Phase 1 复制（不变）
```

---

## 实现顺序

1. **先做第一步**：创建 `dataset_online.py` + 修改 `train_phase1_5.py`（只改数据加载）。跑一个 quick test——加载数据、检查 shape、确认每 epoch 样本量正确。这一步可以单独验证，确认数据多样性提升是否有效。

2. **再做第二步**：在 `mm_adapter_vit.py` 中新增 `LoRALinear` + `inject_lora_into_mm_block`。在 `model_phase1.py` 中新增 `lora_rank`/`lora_alpha` 参数。在 `train_phase1_5.py` 中加入 LoRA 参数分组。

3. **先验证第一步**：用 Phase 1 的 checkpoint + 新数据加载跑 5 epoch，观察 loss 下降和验证 mIoU，确认在线裁剪无 bug 且效果提升。

4. **再联合训练**：Phase 1 checkpoint 转 LoRA（LoRA 初始化为零，不影响已有权重），用新数据 + LoRA 继续训练 10-20 epoch。
