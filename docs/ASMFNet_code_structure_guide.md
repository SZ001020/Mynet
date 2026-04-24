# ASMFNet 代码结构与实现说明

本文档面向当前目录下的 ASMFNet 代码，回答两个核心问题：

- 代码结构是什么样的
- 具体如何实现 RGB + DSM 的多模态语义分割

## 1. 项目定位

ASMFNet 对应论文 Adjacent-Scale Multimodal Fusion Networks for Semantic Segmentation of Remote Sensing Data（JSTARS 2024）。

当前实现的核心思路是：

1. 使用 Swin Transformer U-Net 作为骨干。
2. 为 RGB 与 DSM 建立双分支编码路径。
3. 在相邻尺度之间通过 AMF 模块做跨模态融合。
4. 解码阶段结合 skip connection 还原高分辨率分割图。
5. 采用滑窗推理处理大尺寸遥感影像。

## 2. 目录结构总览

ASMFNet 目录关键文件如下。

- README.md
作用：论文信息与基础运行方式说明。

- train.py
作用：训练入口，含数据集定义、训练循环、验证评估、日志与 checkpoint 保存。

- utils.py
作用：工具函数，包含颜色映射、随机裁剪、损失封装、滑窗推理与指标函数。

- inference_best.py
作用：加载指定 best 模型做平滑推理并导出预测图。

- models/swinfusenet/vision_transformer.py
作用：SwinFuseNet 顶层封装，负责模型构建与预训练权重加载。

- models/swinfusenet/swin_transformer_unet_skip_expand_decoder_sys.py
作用：SwinTransformerSys 主体实现，包含双分支编码、AMF 融合、上采样解码。

- fix_log_metrics.py
作用：从日志里提取混淆矩阵并重新计算指标（含排除 undefined 类的后处理）。

- plot_loss_asmf.py
作用：从训练日志抽取 loss 曲线并绘制图像。

- pretrain/swin_tiny_patch4_window7_224.pth
作用：Swin-Tiny 预训练权重。

- inference_results/
作用：推理输出目录。

## 3. 训练主流程（train.py）

train.py 的执行链路如下。

1. 读取训练超参数
- 包括 seed、窗口大小、学习率、日志目录、是否 AMP 等。
- 默认窗口为 224x224，并检查是否为 28 的倍数（兼容 Swin window=7）。

2. 构建模型
- 实例化 SwinFuseNet。
- 调用 load_from 加载预训练权重。

3. 构建数据集与 DataLoader
- 内置 ISPRS_dataset 类，随机 tile + 随机裁剪 patch。
- 输入包含 RGB patch、DSM patch、label patch。

4. 训练循环
- 前向 output = net(data, dsm)
- 损失 CrossEntropy2d(output, target)
- 支持 AMP 的 GradScaler 反向更新。

5. 周期验证
- 每隔若干 epoch 调用 test 做滑窗推理。
- 计算 OA、F1、Kappa、mIoU 等指标。

6. 日志与模型保存
- 保存标准化 CSV 指标。
- 保存 ASMFNet_best.pth、ASMFNet_last.pth，以及可选 interval checkpoint。

## 4. 数据与评估实现

### 4.1 数据读取

ISPRS_dataset 定义在 train.py 中，主要逻辑：

1. 随机选择一个 tile。
2. 读取 RGB、DSM、label。
3. DSM 执行 min-max 归一化。
4. 随机裁剪固定窗口。
5. 进行翻转与镜像增强。
6. 返回 torch tensor。

### 4.2 损失与指标

在 utils.py 中：

- CrossEntropy2d
对 BxCxHxW 语义分割输出做像素级交叉熵。

- metrics
基于 confusion matrix 计算：
1. Total Accuracy
2. 每类 F1
3. mean F1（前五类）
4. Kappa
5. mean mIoU（前五类）

train.py 会在验证后把指标写入 CSV，便于后续绘图和对比。

### 4.3 滑窗推理

test 函数和 inference_best.py 都采用滑窗策略：

1. 对整图按窗口切块。
2. 分批推理每个 patch。
3. 把 logits 累加回整图位置。
4. 对通道 argmax 得到最终类别图。

这能兼容大图且减轻显存压力。

## 5. 顶层模型封装（vision_transformer.py）

SwinFuseNet 类是训练脚本直接调用的模型接口。

### 5.1 输入处理

forward(x, y) 中：

- 如果 x 是单通道，会复制成 3 通道。
- DSM y 先 unsqueeze 成 1 通道，再 repeat 成 3 通道。
- 两个 3 通道输入共同送入 SwinTransformerSys。

### 5.2 预训练权重加载

load_from(path) 的关键行为：

1. 支持两类 checkpoint 格式（包含 model 字段或不包含）。
2. 对编码器权重做键名扩展：
- patch_embed 同步映射到 DSM 分支 patch_embedd。
- layers 同步映射到 DSM 分支 layersd。
- 同时构造 decoder 对应的 layers_up 权重初始化。
3. 对 shape 不匹配参数自动剔除再加载。

这实现了从单分支 Swin 预训练到双分支融合网络的参数迁移。

## 6. 核心网络结构（swin_transformer_unet_skip_expand_decoder_sys.py）

该文件实现了 ASMFNet 的主体 SwinTransformerSys。

### 6.1 基础模块

- PatchEmbed / PatchMerging / PatchExpand / FinalPatchExpand_X4
用于 token 化、降采样、上采样。

- WindowAttention / SwinTransformerBlock / BasicLayer
实现 Swin 的窗口注意力和分层编码。

- BasicLayer_up
解码端的上采样版 Swin block。

### 6.2 双分支编码

SwinTransformerSys 同时维护两套编码层：

- layers：RGB 分支
- layersd：DSM 分支

forward_features 中每个 stage 都让两分支并行前进，再进入融合模块。

### 6.3 AMF 邻近尺度融合

关键模块是 AMF（Adjacent-scale Multimodal Fusion）：

- 输入当前尺度与相邻尺度的 RGB/DSM 特征。
- 使用 HSA（跨尺度融合）和 ECA（通道注意力）提取信息。
- 通过 AdaptiveFusion 自适应融合 RGB 与 DSM。
- 输出融合 token，作为后续 stage 和解码 skip 的信息源。

代码中通过 layerFu = [AMF(...), AMF(...), AMF(...), AMF(...)] 在四个尺度串联执行。

### 6.4 解码与输出

forward_up_features 中：

1. 从瓶颈特征逐级上采样。
2. 与融合后的多尺度 skip 特征拼接。
3. 线性层对拼接通道回压。
4. 最终 FinalPatchExpand_X4 上采样到原图尺度。
5. 1x1 卷积输出 num_classes 通道 logits。

## 7. 推理脚本（inference_best.py）

该脚本提供独立推理流程：

1. 加载指定 best 模型权重。
2. 对 test_ids 执行平滑推理（默认 stride 64，窗口 224）。
3. 生成类别图并转换伪彩色。
4. 保存到 inference_results 目录。

与 train.py 内 test 相比，这个脚本更偏向直接出图部署。

## 8. 辅助脚本

### 8.1 fix_log_metrics.py

用途：

- 从日志中提取混淆矩阵。
- 可按需求排除 undefined 类后重新统计 OA、F1、mIoU。
- 生成 real_metrics_summary.log。

适合核对“论文口径指标”和“训练日志口径指标”的差异。

### 8.2 plot_loss_asmf.py

用途：

- 解析训练日志中的 Train (...) Loss 行。
- 绘制原始 loss 与平滑 loss 曲线。
- 保存到 asmfnet_loss_curve.png。

## 9. 当前实现特点与注意点

1. 数据集路径有硬编码与环境变量混用
- 推荐统一为配置文件或纯环境变量输入。

2. 数据集类定义在 train.py 内部
- 便于单脚本运行，但复用性较低。

3. 训练中保存了标准 CSV
- 这比旧版项目更利于实验管理和复现。

4. 代码里存在个别注释/字符串小瑕疵
- 例如 CACHE 那行注释尾部有拼接残留文本，不影响主流程但可清理。

## 10. 建议阅读顺序

建议按以下顺序阅读 ASMFNet：

1. train.py
2. models/swinfusenet/vision_transformer.py
3. models/swinfusenet/swin_transformer_unet_skip_expand_decoder_sys.py
4. utils.py
5. inference_best.py
6. fix_log_metrics.py 与 plot_loss_asmf.py

这样可以先抓住训练驱动，再深入双模态 Swin 融合细节，最后看推理与分析工具。

## 11. 一句话总结

ASMFNet 的工程本质是：

- 双分支 Swin 编码（RGB/DSM）
- 邻近尺度跨模态融合（AMF/HSA/ECA/AdaptiveFusion）
- U-Net 式多级解码恢复分割图

它是一个基于 Swin Transformer 的多尺度多模态遥感语义分割框架。
