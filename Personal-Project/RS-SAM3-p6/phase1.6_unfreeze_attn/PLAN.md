# Plan6 Phase 1.6: 选择性解冻深层 ViT Attention

> 目标：在 Phase 1 MMAdapter (75.72%) 基础上，解冻最后 4 层 ViT attention 权重，突破 frozen 天花板
> 预期：78-80% mIoU

## 动机

Phase 1-1.5 总共 5 次实验，256² 整图 mIoU 全部落在 73-76%：

| 实验 | 256² mIoU | 瓶颈 |
|------|-----------|------|
| VPT+MFNet | 73.10% | frozen attention |
| Full SGD 456M | 72.87% | 全部解冻，12 张图喂不饱 |
| MMAdapter | 75.72% | frozen attention |
| MMAdapter+LoRA | 75.57% | LoRA 对 SAM3 帮助微小 |

MFNet Frozen SAM1 = 75.11%。在 frozen 路线下，我们的 MMAdapter 75.72% 已经超过 SAM1 frozen baseline。差距不在 decoder 端，在 ViT 内部。

MFNet Best 84.72% 的核心差异：SAM1 attention 被 LoRA 修改了。SAM3 的 fused QKV + window attention 让 LoRA 低效，那就直接解冻 attention 权重本身——但只解冻最后几层，不碰浅层通用特征。

## 方案

基于 Phase 1 最优配置 + 在线裁剪，增加：

```
32 个 ViT blocks:
  blocks 0-27: 完全 frozen（保留预训练通用特征）
  blocks 28-31: attention 权重解冻（qkv, proj），MLP 保持 frozen
  MMAdapter: 全部 32 层保持 trainable
  Decoder: trainable
```

参数分组与学习率：

| 参数组 | 内容 | LR | 说明 |
|--------|------|-----|------|
| adapter | MMAdapter + DSM encoder | 1e-4 | 与 Phase 1 一致 |
| decoder | MFNetDecoder | 1e-4 | 与 Phase 1 一致 |
| unfrozen_attn | 最后 4 层 qkv+proj | **5e-7** | 极端保守，防止破坏预训练权重 |

总可训练参数：~7M (adapter 5M + decoder 0.5M + unfrozen attn ~1.5M)

## 训练配置

- 数据：在线随机裁剪（Phase 1.5 的 OnlineCropDataset），epoch_steps=2000
- Epochs：30
- Batch：2
- Optimizer：AdamW
- 调度器：CosineAnnealing
- Loss：structure_loss
- DSM 模式：full attention + checkpoint

## 预期

- 保守：77-78%（比 Phase 1 +1-2pp）
- 理想：79-80%（深层 attention 适应遥感后显著提升）

## 文件

```
phase1.6_unfreeze_attn/
├── PLAN.md
├── model_phase1.py        ← 修改：支持 selective unfreeze
├── mm_adapter_vit.py       ← 从 phase1.5 复制，去掉 LoRA 相关代码
├── dataset_online.py       ← 从 phase1.5 复制
├── train_phase1_6.py       ← 新建：三参数组优化
├── mfnet_decoder.py        ← 从 phase1 复制
├── structure_loss.py       ← 从 phase1 复制
├── eval_mfnet_protocol.py  ← 从 phase1.5 复制并修正路径
└── dataset_adapter.py      ← 从 phase1.5 复制
```
