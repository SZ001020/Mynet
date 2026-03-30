# v1.1 代码改造清单（精确到文件与模块）

## 1. 改造目标
- 将 MFNet 扩展为统一训练框架：监督分割 + 多层域对齐 + 边界对象结构约束 + 自动提示伪标签。
- 主线实验以 ISPRS Vaihingen/Potsdam 为核心，LoveDA 仅作为可选压力测试。
- 保持与现有代码兼容，最小侵入式改造。

## 2. 新增文件（建议）

### 2.1 新增训练入口
- 文件：MFNet/train_uda_struct_v1.py
- 职责：统一管理监督损失、对抗损失、边界损失、对象损失、伪标签掩码与提示损失。
- 需新增模块
  - Trainer 类
  - build_optimizer 函数（主干与判别器分组）
  - validate_source 与 validate_target 函数
  - run_protocol 函数（same-domain 和 weak-cross-domain 两种协议）
  - dump_artifacts 函数（统一导出日志、图像、表格原始数据）

### 2.2 新增域判别器
- 文件：MFNet/modules/domain_discriminator.py
- 职责：对多层语义特征进行域二分类（默认高层，可选中层+高层）。
- 需新增模块
  - GradientReversalLayer
  - PixelDomainDiscriminator（轻量卷积版本）
  - MultiLevelDomainHead（可选，接收层级字典）

### 2.3 新增结构损失
- 文件：MFNet/modules/structure_losses.py
- 职责：封装边界损失与对象一致性损失，兼容 target 伪标签掩码。
- 需新增模块
  - BoundaryLossWrapper
  - ObjectLossWrapper
  - masked_loss 函数

### 2.4 新增伪标签工具
- 文件：MFNet/modules/pseudo_labeling.py
- 职责：生成 target 高置信伪标签与掩码，并支持提示约束融合。
- 需新增模块
  - get_pseudo_label
  - confidence_mask
  - class_balanced_threshold（可选）
  - prompt_guided_refine（用提示结果修正伪标签）

### 2.5 新增自动提示工具
- 文件：MFNet/modules/prompt_generator.py
- 职责：从 target 高置信区域生成点/框提示，供 SAM 分支或提示一致性损失使用。
- 需新增模块
  - generate_point_prompts
  - generate_box_prompts
  - sample_prompt_set（控制每类提示数量）

### 2.6 新增配置文件
- 文件：MFNet/configs/uda_struct_v1.yaml
- 职责：集中配置数据路径、损失权重、训练轮数、阈值、日志路径与提示策略。
- 建议新增字段
  - lambda_prompt
  - adv_levels: [high] 或 [mid, high]
  - prompt_mode: point | box | hybrid
  - artifact_dump_interval

### 2.7 新增数据协议配置
- 文件：MFNet/configs/protocol_isprs.yaml
- 职责：定义主线数据协议与划分。
- 建议字段
  - source_domain: Vaihingen
  - target_domain: Potsdam
  - mode: same_domain | weak_cross_domain
  - train_ids / val_ids / test_ids

### 2.8 新增实验产物规范配置
- 文件：MFNet/configs/artifact_spec_v1_1.yaml
- 职责：定义每次实验必须导出的数据文件、图片文件与表格文件模板。
- 建议字段
  - save_csv: true
  - save_per_class_iou: true
  - save_pseudo_stats: true
  - save_visual_cases: true
  - figure_templates / table_templates

## 3. 修改现有文件（必改）

### 3.1 MFNet/UNetFormer_MMSAM.py
- 改造点
  - 在 UNetFormer.forward 中新增可选返回特征：return_feat=True 时返回 seg_logits 与特征字典 feat_dict。
  - 增加 forward_target 分支复用主干（保持参数共享）。
- 新增接口
  - forward(self, x, y, mode='Train', return_feat=False, feat_levels=('high',))
- 目的
  - 让训练脚本能同时拿到分割输出与多层域对齐特征。

### 3.2 MFNet/train.py
- 改造点
  - 保留原脚本用于纯监督训练。
  - 增加提示导向到 train_uda_struct_v1.py，避免逻辑过度堆叠。
- 目的
  - 不破坏原可复现流程。

### 3.3 MFNet/utils.py
- 改造点
  - 增加 target 域数据读取接口（同域有标签 / 弱跨域无标签两套模式）。
  - 增加多 dataloader 迭代器同步函数（source 与 target 同步采样）。
- 需新增函数
  - build_source_target_loaders
  - infinite_loader
  - build_same_domain_loaders
  - build_weak_cross_domain_loaders

## 4. 可复用外部现有代码（直接迁移）

### 4.1 来自 GLGAN 的对抗训练思路
- 参考文件：GLGAN/FDGLGAN_LoveDA_R2U.py
- 迁移内容
  - 域判别器训练节奏。
  - 主干与判别器交替更新策略。
  - 数据集无关化封装（去掉 LoveDA 强绑定路径与类别假设）。

### 4.2 来自 SAM_RS 的结构损失
- 参考文件：SAM_RS/train.py
- 迁移内容
  - BoundaryLoss 与 ObjectLoss 调用方式。
  - 与分割 CE 损失的加权组合。

## 5. 第一版损失函数组合（建议默认）
- 总损失
  - L_total = L_seg_src + lambda_adv * sum(L_adv_l) + lambda_bdy * L_bdy + lambda_obj * L_obj + lambda_prompt * L_prompt
- 建议初值
  - lambda_adv = 0.001
  - lambda_bdy = 0.1
  - lambda_obj = 1.0
  - lambda_prompt = 0.05
- 伪标签阈值
  - tau = 0.85（初始）

## 5.1 主线训练协议（非 LoveDA）
- 协议 A：Same-domain
  - source 与 target 来自同一数据集同一域。
  - 启用 L_seg_src + L_bdy + L_obj + L_prompt，不启用 L_adv。
- 协议 B：Weak-cross-domain
  - source 为 Vaihingen，target 为 Potsdam（或反向）。
  - 启用 L_seg_src + L_adv + L_bdy + L_obj + L_prompt。
- 协议 C：Optional stress
  - 可选 LoveDA，仅用于压力测试与泛化展示。

## 6. 训练流程改造（按步骤）
1. 读取 source 有标签 batch 与 target 无标签 batch。
2. 前向 source，计算 L_seg_src。
3. 前向 target，生成伪标签与 mask，并生成点/框提示。
4. 用 source+target 多层特征计算 L_adv。
5. 在 target mask 区域上计算 L_bdy 与 L_obj，并计算 L_prompt。
6. 反向传播更新主干。
7. 再单独更新域判别器。

注：same-domain 协议下跳过第 4 步和第 7 步。

## 7. 日志与结果文件约定
- 建议新增目录
  - MFNet/runs/uda_struct_v1/
  - MFNet/checkpoints/uda_struct_v1/
  - MFNet/results/uda_struct_v1/
- 记录项
  - 每 iter：L_seg、L_adv_l、L_bdy、L_obj、L_prompt、伪标签覆盖率
  - 每 epoch：source mIoU、target mIoU、Boundary F1、域判别器准确率

## 7.1 必须保存的数据清单（v1.1）
- 配置与环境：完整配置、命令、git commit、随机种子、硬件信息。
- 训练中间数据：loss 与 metric 原始 csv、每类 IoU 明细、伪标签统计。
- 模型状态：best/last checkpoint + optimizer/scheduler 状态。
- 推理结果：原图、预测图、标签图、误差图、边界图。

## 7.2 必须生成的图片（v1.1）
- 训练损失曲线图（全部损失项）。
- 主指标曲线图（mIoU、Boundary F1）。
- 伪标签置信度热力图（至少 early/mid/late 三阶段）。
- 可视化对比图（baseline vs +adv vs +adv+bdy+obj vs +prompt）。

## 7.3 必须生成的表格（v1.1）
- 主结果表：三协议下各模型主指标对比。
- 消融表：去掉各损失项与多层对齐/单层对齐对比。
- 参数效率表：全量微调、LoRA、Adapter 的参数量、显存、速度、精度。
- 稳定性表：3 seeds 均值与标准差。

## 8. 第一版验收清单（完成即通过）
- 能完整跑通 1 次 same-domain 训练与 1 次 weak-cross-domain 训练（无报错）。
- 能保存最优 checkpoint 与可视化预测图。
- 相比 MFNet 纯监督基线，weak-cross-domain target mIoU 有正向提升。
- 能输出 1 张消融对照（至少 -adv 或 -bdy）。
- 能输出 1 组提示机制消融（固定阈值伪标签 vs 自动提示伪标签）。

## 9. 第二版预留接口（本版先留钩子）
- 类别级域对齐（class-wise adversarial alignment）。
- 频域分解分支（借鉴 FDGLGAN）。
- 不确定性加权伪标签（entropy-aware weighting）。
