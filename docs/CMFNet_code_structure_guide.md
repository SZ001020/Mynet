# CMFNet 代码结构与实现说明

本文档面向当前目录下的 CMFNet 代码，说明两件事：

- 代码结构是什么样的
- 具体如何实现 RGB + DSM 的多模态语义分割

## 1. 项目定位

CMFNet 对应论文 A Crossmodal Multiscale Fusion Network for Semantic Segmentation of Remote Sensing Data（JSTARS 2022）。

该实现的核心思想是：

1. 使用双编码器分别提取 RGB 与 DSM 特征。
2. 用通道注意力和跨尺度 Transformer 在中高层做跨模态融合。
3. 用 SegNet 风格的池化索引反卷积解码，恢复像素级预测。
4. 通过滑窗推理对大图进行评估与可视化输出。

## 2. 目录结构总览

CMFNet 目录中的关键文件如下。

- README.md
作用：论文与基础运行方式说明。

- Config.py
作用：定义 CTranS 相关超参数（层数、头数、patch size、KV 大小等）。

- trainAndtest.py
作用：训练和测试主脚本，包含数据集定义、训练循环、评估函数、权重初始化与保存。

- inference.py
作用：独立推理脚本，加载训练后权重并导出整图分割结果。

- Utils/CMFNet.py
作用：主模型定义。包含双分支编码器、跨模态跨尺度 Transformer、注意力融合模块、解码器。

- Utils/utils.py
作用：通用工具函数。包含颜色映射、随机裁剪、交叉熵封装、滑窗、评估指标。

## 3. 训练与推理主流程

### 3.1 trainAndtest.py 主流程

训练脚本的执行链路如下：

1. 定义数据路径和超参数（窗口大小、批大小、类别数等）。
2. 构建 ISPRS_dataset，随机采样 tile 并随机裁剪 patch。
3. 初始化 CMFNet 模型。
4. 加载 VGG16-BN 预训练权重并映射到编码器。
5. 使用 SGD 训练（带 MultiStepLR）。
6. 每隔若干 epoch 调用 test 进行验证。
7. 按验证精度保存 best 权重，同时保存 last 权重和 mean_losses。

### 3.2 test 评估流程

test 函数采用滑动窗口对整张图推理：

1. 对每个测试 tile 读取 RGB、DSM、label。
2. 对 RGB/DSM 分块并 batch 推理。
3. 把每个块的 logits 累加回原图位置。
4. 对通道做 argmax 得到类别图。
5. 汇总全测试集预测后用 metrics 计算 OA、F1、Kappa、mIoU。

### 3.3 inference.py 推理流程

inference.py 是简化版部署脚本：

- 读取训练后的模型参数。
- 对 test_ids 执行滑窗推理。
- 把预测类别图转换成伪彩色结果并保存到输出目录。

与 trainAndtest.py 相比，inference.py 更偏向批量出图使用。

## 4. 主模型结构（Utils/CMFNet.py）

主类是 CMFNet，整体可以理解为：

- 双模态编码器（RGB 编码器 + DSM 编码器）
- 多级融合模块（通道注意力 + CTranS）
- SegNet 风格解码器

### 4.1 双编码器

CMFNet 在编码阶段有两套卷积块：

- RGB 编码器：conv1_1 到 conv5_3
- DSM 编码器：conv1_1_d 到 conv5_3_d

输入处理方式：

- RGB 输入保持 3 通道。
- DSM 输入先 unsqueeze 成 1 通道后进入深度分支。

两分支都进行多次卷积 + BN + ReLU，并记录多层特征（x1~x4、y1~y4）。

### 4.2 通道注意力融合

模型内置 attention(num_channels) 生成 squeeze-excitation 风格通道注意力：

- 对特征做全局池化
- 1x1 卷积映射
- Sigmoid 得到通道权重

在第五层特征上，RGB 与 DSM 各自先做通道加权，再相加得到融合特征 x5。

### 4.3 CTranS 跨模态跨尺度模块

模型中有两个 Transformer 融合器：

- mtc: ChannelTransformer_cross
用途：同时输入 RGB 与 DSM 的四个尺度特征，做跨模态交互。

- mtc1: ChannelTransformer
用途：在融合后的 RGB 特征上再做一次跨尺度通道 Transformer 建模。

这两步输出的 xtf1~xtf4 被送入解码端作为增强 skip 特征。

### 4.4 CTranS 子模块组成

CTranS 相关类包括：

- Channel_Embeddings
把各尺度特征按 patch 切分并加位置编码。

- Attention_org / Attention_org_cross
分别用于单流与跨流注意力。

- Block_ViT / Block_ViT_cross
LayerNorm + 注意力 + MLP + 残差的 Transformer block。

- Encoder / Encoder_cross
多层堆叠 Block，输出编码后的 token。

- Reconstruct
把 token 重排为特征图并上采样回对应尺度。

- ChannelTransformer / ChannelTransformer_cross
完整封装了 embedding -> encoder -> reconstruct 的流程，并加残差回注。

### 4.5 解码器

解码部分采用 SegNet 风格：

- 编码时 MaxPool2d(return_indices=True) 保存池化索引。
- 解码时 MaxUnpool2d 逐层恢复分辨率。
- 每层与对应的 x 特征以及 xtf 特征相加再卷积。
- 最后一层输出类别通道并执行 log_softmax。

输出张量形状为 B x C x H x W，C 是类别数。

## 5. 配置实现（Config.py）

Config.py 中 get_CTranS_config 定义了 CTranS 的关键超参数：

- KV_size = 960
- num_heads = 4
- num_layers = 4
- expand_ratio = 4
- patch_sizes = [16, 8, 4, 2]
- base_channel = 64

这些参数主要驱动 ChannelTransformer 与 ChannelTransformer_cross 的建模容量。

## 6. 数据与损失实现（Utils/utils.py + trainAndtest.py）

### 6.1 数据读取

数据集类 ISPRS_dataset 定义在 trainAndtest.py 中，逻辑是：

1. 随机选择一张 tile。
2. 读取 RGB、DSM、GT。
3. 对 DSM 做 min-max 归一化。
4. 随机裁剪 256x256 patch。
5. 随机翻转与镜像增强。
6. 返回 RGB patch、DSM patch、label patch。

### 6.2 损失函数

训练使用 CrossEntropy2d（在 Utils/utils.py 中封装）。

模型输出是 log_softmax，训练时直接按像素分类优化。

### 6.3 指标函数

metrics 返回并打印：

- Confusion matrix
- Overall Accuracy
- 每类 Accuracy/F1
- mean F1（前景类）
- Kappa
- mean IoU（前景类）

注意：trainAndtest.py 中用于 best 模型判断的 acc 是 metrics 返回的 OA。

## 7. 预训练权重初始化策略

trainAndtest.py 会下载并加载 VGG16-BN 权重，然后映射到 CMFNet：

1. 把 VGG 的 features 权重按顺序映射到模型对应层。
2. 对 DSM 首层 conv1_1_d.weight，使用 RGB 首层卷积的通道均值进行 1 通道初始化。
3. 其余带 _d 的层，若存在同名 RGB 层参数，则复用对应权重。

这个策略能让 DSM 分支继承 RGB 分支的低层先验，稳定初期训练。

## 8. 当前实现特点与注意点

1. 路径与数据集配置较硬编码
- 例如 FOLDER 与 DATASET 默认写死在脚本中。
- 建议改为环境变量或 yaml 配置，提高复现性。

2. trainAndtest.py 训练测试耦合在同一脚本
- 便于快速实验，但工程化复用性一般。

3. inference.py 是独立推理脚本
- 便于部署，但部分逻辑与 trainAndtest.py 存在重复。

4. 日志体系较轻
- 当前以打印和 mean_losses.npy 为主。
- 可进一步对齐你现在使用的 CSV 标准日志格式。

## 9. 建议阅读顺序

建议按以下顺序阅读 CMFNet：

1. trainAndtest.py
2. Utils/CMFNet.py
3. Config.py
4. Utils/utils.py
5. inference.py

这样可以先抓住训练驱动，再看模型细节，最后看推理落地。

## 10. 一句话总结

CMFNet 的工程本质是：

- 双分支 CNN 编码提取 RGB/DSM 表征
- 通过 CTranS 在多尺度上做跨模态 token 级交互
- 再用 SegNet 式解码与 skip 融合恢复分割结果

它是一个典型的 CNN + Transformer 混合多模态语义分割框架。
