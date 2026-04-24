# FTransUNet 代码结构与实现说明

本文档面向当前目录下的 FTransUNet 代码，目标是回答两个问题：

- 代码结构是什么样的
- 具体是怎么实现多模态遥感语义分割的

## 1. 项目定位

FTransUNet 对应论文 A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation。

核心思想是：

1. 用 RGB 与 DSM 两种模态作为输入。
2. 在 ResNet+Transformer 主干中进行多层次融合。
3. 在解码端利用 skip connection 进行高分辨率重建。
4. 输出语义分割 logits，计算像素级交叉熵进行训练。

## 2. 目录结构总览

当前 FTransUNet 目录中的关键文件如下。

- README.md
作用：论文信息与基础运行说明。

- train.py
作用：主训练入口，包含训练循环、验证评估、日志记录、权重保存。

- utils.py
作用：数据集协议、数据读取与增强、损失函数、滑窗推理工具、指标计算。

- model/vitcross_seg_modeling.py
作用：核心模型定义。包含融合 ResNet 主干、双分支 Transformer 编码器、解码器、预训练权重加载逻辑。

- model/vit_seg_modeling_resnet_skip.py
作用：ResNetV2 与 FuseResNetV2。实现 RGB/DSM 早期卷积与多尺度融合，输出给 Transformer 与解码器跳连分支。

- model/vit_seg_configs.py
作用：模型配置注册（例如 R50-ViT-B_16），定义 Transformer 层数、隐藏维度、预训练权重路径、解码器通道等。

- model/vitcross_seg_modeling_heatmap.py
作用：热力图分析版本模型，在 forward 内额外计算并返回 3 个层级的 Grad-CAM 风格 heatmap。

- test_heatmap.py
作用：可视化脚本，调用 heatmap 版模型并导出可视化图。

- predict.py
作用：加载指定权重并批量导出彩色分割图。

- plot_ftransunet_loss.py
作用：从训练日志提取 Loss/Accuracy，画训练曲线。

- pretrain/R50+ViT-B_16.npz
作用：ViT+ResNet 预训练权重来源。

## 3. 端到端执行链路

### 3.1 训练主流程

入口是 train.py。

1. 环境与随机种子初始化
- 读取 SSRS_SEED、SSRS_PERF_MODE、SSRS_LOG_DIR 等环境变量。
- 固定随机种子以增强复现性。

2. 构建模型
- 从 CONFIGS_ViT_seg 选 R50-ViT-B_16。
- 设置 n_classes=6，n_skip=3，patch grid 为 16x16（输入 256 时）。
- 实例化 VisionTransformer，并调用 load_from 加载 npz 预训练权重。

3. 数据加载
- 使用 ISPRS_dataset 构建 DataLoader。
- 输入为 RGB patch 与 DSM patch，标签为语义类别图。

4. 训练循环
- 前向：output = net(data, dsm)
- 损失：CrossEntropy2d(output, target)
- 反向：loss.backward + optimizer.step

5. 周期验证
- 调用 test 函数进行滑窗推理和整图评估。
- metrics 输出混淆矩阵、总精度、各类 F1、Kappa、mean mIoU。

6. 日志与模型保存
- 将关键指标写入 CSV。
- 保存 best 与 last 权重。

### 3.2 推理流程

入口可用 train.py 末尾测试段或 predict.py。

- 加载模型权重。
- 对 test_ids 做滑窗推理，窗口预测结果累加到整图概率图。
- argmax 得到分割类别图。
- convert_to_color 转成 ISPRS 配色并保存为 png。

## 4. 模型实现细节

### 4.1 总体结构

核心类在 model/vitcross_seg_modeling.py 的 VisionTransformer。

前向可概括为：

1. Transformer 编码（双分支）
- x 分支处理 RGB
- y 分支处理 DSM

2. 编码后融合
- x = x + y

3. 解码
- DecoderCup + skip features 逐层上采样

4. 输出
- SegmentationHead 输出类别 logits

### 4.2 多模态融合主干：FuseResNetV2

在 model/vit_seg_modeling_resnet_skip.py 中：

- RGB 走 root（3 通道），DSM 走 rootd（1 通道）。
- 两支分别经过对称 ResNet body/bodyd。
- 每个阶段通过 SqueezeAndExciteFusionAdd 做融合（SE 注意力后相加）。
- 输出：
1. 高层融合特征 x、y（供 Transformer patch embedding）
2. 多尺度 features（供解码器 skip connection）

这是一种先 CNN 多尺度融合，再 Transformer 建模全局关系的路线。

### 4.3 Transformer 编码器

由 Embeddings、Encoder、Block、Attention、Mlp 组成。

- Embeddings
1. 如果配置是 R50 hybrid，会先调用 FuseResNetV2。
2. 再分别用 patch_embeddings 和 patch_embeddingsd 把 x/y 投影成 token。
3. 添加共享位置编码 position_embeddings。

- Encoder
1. 堆叠 12 层 Block。
2. 层模式为 3+6+3：前 3 与后 3 是 SA，中间 6 是 MBA。

- Block
1. LayerNorm
2. Attention
3. 残差
4. MLP
5. 残差

### 4.4 SA 与 MBA 的实现

Attention 模块里同时维护 x/y 两组 QKV。

- SA（self attention）
- x 用 Qx,Kx,Vx
- y 用 Qy,Ky,Vy

- MBA（multimodal bi-attention）
在 SA 之外再计算交叉注意力：
- x 用 Qx,Ky,Vy
- y 用 Qy,Kx,Vx

然后用可学习系数融合：

- attention_sx = w11 * SA_x + w12 * Cross_x
- attention_sy = w21 * SA_y + w22 * Cross_y

这使模型可以在层内自适应调节单模态建模与跨模态交互的占比。

### 4.5 解码器

DecoderCup 将 token 恢复为空间特征图，再通过 4 个 DecoderBlock 上采样。

- 输入 hidden states reshape 成 B,C,H,W。
- conv_more 做通道适配。
- 每层可拼接一个 skip feature。
- 最终 SegmentationHead 输出类别图。

## 5. 预训练权重加载机制

VisionTransformer.load_from 负责把 npz 权重映射到当前模型。

关键点：

1. patch embedding 权重同时拷贝到 x/y 两个分支。
2. Transformer encoder block 权重逐层拷贝。
3. 位置编码支持尺寸不一致时插值 resize。
4. 如果 hybrid=True，还会加载 ResNet root 与各个 bottleneck 的参数。

其中 DSM rootd 的第一层卷积由 RGB 卷积权重按通道均值初始化，是从 3 通道迁移到 1 通道的常见做法。

## 6. 数据与评估实现

### 6.1 数据协议

utils.py 中按 SSRS_DATASET 切换：

- Vaihingen
- Potsdam

并定义：

- train_ids / test_ids
- RGB/DSM/label/eroded label 路径模板
- Stride_Size

### 6.2 数据集类 ISPRS_dataset

样本生成逻辑：

1. 随机选一张 tile。
2. 读取 RGB、DSM、label（支持 cache）。
3. 随机裁剪 WINDOW_SIZE patch。
4. 随机翻转/镜像增强。
5. 返回 data_p, dsm_p, label_p。

### 6.3 损失与指标

- 训练损失：CrossEntropy2d
- 评估：metrics
输出包含：
- confusion matrix
- total accuracy
- 每类 F1
- mean F1
- Kappa
- mean mIoU

注意：train.py 中模型选择 best 的指标变量名是 acc，本质取自 metrics 的返回值（该实现返回 overall accuracy）。日志中同时记录了 mean_miou 等详情。

## 7. 热力图实现

### 7.1 heatmap 模型

model/vitcross_seg_modeling_heatmap.py 在 forward 内保留三个层级特征：

1. cnn_x
2. trans_x
3. decoder 输出 x

随后对某个像素类别 logit（示例里是 pred = logits[:, 3, 100, 65]）求相对这些特征的梯度，计算 Grad-CAM 风格 heatmap。

返回值为：

- logits
- heatmap1
- heatmap2
- heatmap3

### 7.2 可视化脚本

test_heatmap.py 会把 RGB patch、三个 heatmap、GT 并排保存成 pdf，用于解释模型关注区域。

## 8. 训练日志与可视化

- train.py 会写标准化 CSV，便于后续做实验对比。
- plot_ftransunet_loss.py 从文本日志正则提取 Loss/Accuracy 并绘图。

你给出的曲线图就对应这种日志后处理方式：

- 红线是 loss 下降
- 蓝线是 accuracy 上升并趋稳

## 9. 建议阅读顺序

建议按这个顺序读代码：

1. train.py
2. utils.py
3. model/vitcross_seg_modeling.py
4. model/vit_seg_modeling_resnet_skip.py
5. model/vit_seg_configs.py
6. predict.py 与 test_heatmap.py

这样先建立训练驱动逻辑，再深入模型融合细节，最后看部署与解释性输出。

## 10. 一句话总结

FTransUNet 的工程本质是：

- FuseResNetV2 在 CNN 阶段做 RGB-DSM 多尺度融合
- 双分支 Transformer 用 SA + MBA 做 token 级跨模态交互
- UNet 风格解码器利用 skip 恢复细粒度空间信息
- 通过滑窗推理与 ISPRS 指标体系完成训练评估闭环
