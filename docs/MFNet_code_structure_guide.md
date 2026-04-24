# MFNet Code Structure And Implementation Guide

本文档面向当前目录的 MFNet 代码，说明项目结构、关键模块职责、训练流程和实现细节。

## 1. 项目定位

MFNet 是一个面向遥感语义分割的多模态框架，核心思路是：

- 使用 SAM/Image Encoder 作为共享强表征主干。
- 输入 RGB 与 DSM 两种模态，做特征级融合。
- 使用 UNetFormer 风格解码器输出分割结果。
- 训练上支持两条主线：
1. 同域监督训练（`train.py`）
2. 弱跨域 UDA 训练（`train_uda_struct_v1.py`）

主仓内 `MedSAM/` 是深度依赖子模块，提供 SAM 注册器、编码器结构及 LoRA/Adapter 相关实现。

## 2. 目录结构总览

下面是 MFNet 目录中与主流程相关的核心文件：

- `README.md`：论文和基础用法说明。
- `train.py`：同域监督训练入口（week1 baseline）。
- `train_uda_struct_v1.py`：弱跨域 UDA 训练入口（week3 prototype）。
- `UNetFormer_MMSAM.py`：MFNet 主模型定义（编码器、融合、解码器）。
- `UNetFormer_MMSAM_heatmap.py`：热力图分析版模型。
- `test_heatmap.py`：可视化 attention/feature 的测试脚本。
- `utils.py`：数据集读取、数据增强、损失函数、指标评估、滑窗工具。
- `modules/domain_discriminator.py`：域判别器与 GRL（梯度反转层）。
- `configs/protocol_isprs.yaml`：ISPRS 划分协议（train/test IDs, stride）。
- `configs/uda_struct_v1.yaml`：UDA 实验示例配置。
- `scripts/process_hunan.py`：数据预处理脚本草稿（当前基本为注释实验片段）。
- `weights/sam_vit_l_0b3195.pth`：SAM 预训练权重。

此外，`MedSAM/` 目录包含大量 SAM 相关原始与改造代码，是 `UNetFormer_MMSAM.py` 的依赖基础。

## 3. 主模型如何实现（UNetFormer_MMSAM.py）

### 3.1 输入与主干

`UNetFormer` 的前向接口是：

- `x`：RGB 输入，形状一般为 `B x 3 x H x W`
- `y`：DSM 输入，原始通常为 `B x H x W`，前向中会扩展成 3 通道

关键逻辑：

1. `y` 被 `unsqueeze + repeat` 扩展到 3 通道。
2. 调用 `self.image_encoder(x, y)`，得到 RGB 分支与 DSM 分支深层特征（`deepx`, `deepy`）。
3. 两个分支分别过 FPN 风格的多尺度投影（`fpn1x..fpn4x`、`fpn1y..fpn4y`）。
4. 每个尺度通过 `SEFusion` 做通道注意力后融合，得到 `res1..res4`。
5. 将融合特征送入 `Decoder` 输出语义分割 logits。

### 3.2 参数冻结策略

构造函数中会遍历 `image_encoder` 参数：

- 名称不含 `lora_` 的参数默认冻结（`requires_grad=False`）。
- `lora_` 参数保留可训练。

这对应一种 PEFT（参数高效微调）思路：大部分 SAM 编码器权重保持冻结，仅训练 LoRA 相关参数与下游解码融合模块。

### 3.3 解码器

`Decoder` 由以下模块组成：

- `Block`：Global-Local Attention + MLP。
- `WF`：上采样后与浅层特征做可学习权重融合。
- `FeatureRefinementHead`：通道注意力+空间注意力做细化。
- `segmentation_head`：最终输出类别通道。

整体是“多尺度融合 + 细化 + 上采样到输入分辨率”的标准语义分割解码流程。

### 3.4 UDA 支持接口

前向函数支持 `return_feat=True`，可返回特征字典：

- `mid`：中层特征（当前从 `res3`）
- `high`：高层特征（当前从 `res4`）

`train_uda_struct_v1.py` 使用 `high` 特征对接域判别器进行对抗对齐。

## 4. 监督训练流程（train.py）

`train.py` 是当前 MFNet 的同域监督训练主入口。

### 4.1 初始化

主要步骤：

1. 读取环境变量（seed、batch、worker、loss 配置、日志目录等）。
2. `set_seed()` 固定随机性。
3. 构建模型 `MFNet(num_classes=N_CLASSES)`。
4. 构建 `ISPRS_dataset` 和 DataLoader。
5. 设置优化器 `SGD` 与 `MultiStepLR`。
6. 创建 CSV 日志（标准字段包含 loss、metric、lr、结构损失信息）。

### 4.2 数据读取与增强

由 `utils.py` 的 `ISPRS_dataset` 提供：

- 从 `utils.py` 全局配置中解析数据路径模板和 tile ID 列表。
- 支持 RGB 与 DSM 同步读取。
- 支持边界先验图（boundary）和实例图（object）读取。
- 支持随机裁剪与翻转镜像增强。
- `object_process()` 会把实例 ID 重编号为连续整数，便于 object loss 计算。

### 4.3 损失设计

核心包含三项：

- `loss_ce`：主分割交叉熵（`loss_calc`）。
- `loss_boundary`：边界约束（`BoundaryLoss`）。
- `loss_object`：对象一致性约束（`ObjectLoss`）。

支持通过 `SSRS_LOSS_MODE` 切换：

- `SEG`
- `SEG+BDY`
- `SEG+OBJ`
- `SEG+BDY+OBJ`

并支持：

- 结构损失 warmup（`SSRS_STRUCTURE_WARMUP_EPOCHS`）。
- 基于预测置信度阈值的结构损失 gating（`SSRS_STRUCTURE_CONF_THRESH`）。

即：先算分割，再按高置信区域引导 boundary/object 监督，降低伪先验噪声影响。

### 4.4 验证与保存

验证使用滑窗推理：

- `sliding_window` + `grouper` 切 patch。
- 全图 logits 累加后 `argmax`。
- 用 `metrics` 或 `metrics_loveda` 计算 mIoU/F1/Kappa/Acc。

保存逻辑：

- best checkpoint（按 mIoU）。
- last checkpoint。
- 可选 interval checkpoint。
- 每 epoch 同步记录 CSV。

## 5. 弱跨域 UDA 流程（train_uda_struct_v1.py）

这个脚本实现 source->target 的高层特征对抗对齐。

### 5.1 数据组织

- `SourceDataset`：读取源域 RGB/DSM/label/boundary/object。
- `TargetDataset`：读取目标域 RGB/DSM（无标签）。
- `get_protocol()`：根据 `Vaihingen` 或 `Potsdam` 返回路径模板与 split。

### 5.2 域对抗模块

域对抗头在 `modules/domain_discriminator.py`：

- `GradientReversalLayer`：反转梯度，系数 `lambda` 可配。
- `PixelDomainDiscriminator`：卷积判别器，输出逐像素域 logits。
- `DomainAdversarialHead.domain_loss()`：BCEWithLogits 域分类损失。

### 5.3 训练时序（每个迭代）

1. 源域前向，计算 `loss_seg`（可附加 boundary/object）。
2. 目标域前向，拿到高层特征。
3. 使用 GRL 计算 `loss_adv_feat`，反向推动 backbone 学到域不变特征。
4. 单独更新判别器参数，计算 `loss_adv_disc`。

总损失形式：

`total_loss = loss_seg + lambda_adv * loss_adv_feat + lambda_bdy * loss_bdy + lambda_obj * loss_obj`

其中 `lambda_bdy/lambda_obj` 同样支持 warmup 与置信度 gating。

### 5.4 评估与日志

- 周期性在 target test split 上做滑窗评估。
- 记录 target mIoU/F1/Acc 和各类 train loss 到 CSV。
- 保存 `UNetformer_week3_best.pth` 与 `UNetformer_week3_last.pth`。

## 6. 工具与配置

### 6.1 utils.py 的角色

`utils.py` 同时承担：

- 数据集协议和路径模板（通过环境变量可覆盖）。
- 数据读取和增强。
- 损失函数（CE、BoundaryLoss、ObjectLoss）。
- 评估函数（混淆矩阵、F1、Kappa、mIoU）。
- 滑窗推理工具函数。

这是一个“高聚合工具文件”，维护时建议按职责拆分为 data/loss/metrics/config 子模块。

### 6.2 yaml 配置文件现状

- `configs/protocol_isprs.yaml` 与 `configs/uda_struct_v1.yaml` 当前更像实验说明模板。
- 训练脚本主要通过环境变量读取参数，并没有系统地解析这两个 yaml。

如果你后续做可复现实验框架，建议把环境变量统一落到配置解析器，减少参数分散问题。

## 7. 运行链路速览

### 7.1 同域监督训练

1. 启动 `train.py`
2. 读取 `utils.py` 中 dataset 协议和路径
3. DataLoader 采样 RGB/DSM/label(+boundary/object)
4. `UNetFormer_MMSAM` 前向得到 logits
5. 计算 CE + 结构损失
6. 反向更新（主要是 LoRA + 解码融合部分）
7. 滑窗验证并保存 best/last

### 7.2 弱跨域 UDA 训练

1. 启动 `train_uda_struct_v1.py`
2. 构建 Source/Target 双数据流
3. 源域监督分割损失训练
4. 源/目标高层特征送入域判别器进行对抗
5. 周期评估 target，按 mIoU 选 best

## 8. 建议的阅读顺序

建议按如下顺序读代码：

1. `train.py`
2. `utils.py`
3. `UNetFormer_MMSAM.py`
4. `train_uda_struct_v1.py`
5. `modules/domain_discriminator.py`
6. `MedSAM/models/sam/build_sam.py`

这样可以先建立“训练驱动逻辑”，再回到模型细节和跨域机制。

## 9. 一句话总结

MFNet 在工程上可以理解为：

- SAM 编码器 + 多模态融合解码器 的分割主干
- 加上结构先验损失（boundary/object）
- 再扩展到基于高层特征的对抗式跨域适配

因此它既是一个遥感多模态分割框架，也是一个可继续扩展 domain adaptation 的实验基座。
