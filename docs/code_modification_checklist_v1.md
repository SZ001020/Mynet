# 代码改造清单（当前版，结合 v1.1 与参考论文）

## 1. 改造目标
- 将 MFNet 扩展为统一训练框架：监督分割 + weak-cross-domain 域对齐 + 边界对象结构约束。
- 在主线稳定后，再条件性扩展到 multi-level 对齐和 prompt-guided 伪标签精炼。
- 主线实验以 ISPRS Vaihingen/Potsdam 为核心，LoveDA 仅作为可选压力测试。
- 保持与现有代码兼容，采用最小侵入式改造。
- 执行转向（第 6 周起）：同域机制闭环优先，跨域新增实验后置。

## 2. 实现原则

### 2.1 分阶段实现，避免一次性堆满功能
推荐按以下顺序开发：

1. 阶段 A
   先建立统一训练入口、日志规范与 artifact 导出，不改原有监督训练脚本行为。

2. 阶段 B
   实现主线 M0：single-level high-level UDA + 结构约束。

3. 阶段 C
   实现条件增强 E1：multi-level UDA，仅在阶段 B 稳定后接入。

4. 阶段 D
   实现条件增强 E2：prompt-guided 伪标签精炼，仅在阶段 B 或 C 稳定后接入。

第 6 周起执行优先级补充：
1. 先完成 same-domain 的多 seed 机制证据与组件去留判断。
2. 再将通过门槛的组件回灌到 weak-cross-domain。
3. 未通过门槛的组件不进入跨域主线。

### 2.2 代码目标必须服务文档主线
代码实现不应默认把所有扩展都打开。正式默认配置建议只启用：
- source 监督分割；
- high-level 对抗对齐；
- boundary/object 结构约束。

multi-level 和 prompt 分支都应由配置显式开启。

## 3. 建议新增文件

### 3.1 新增统一训练入口
- 文件：MFNet/train_uda_struct_v1.py
- 职责：统一管理监督损失、域对齐损失、边界损失、对象损失、伪标签掩码和可选 prompt 损失。
- 建议新增模块
  - Trainer 类
  - build_optimizer 函数
  - validate_source 与 validate_target 函数
  - run_protocol 函数
  - dump_artifacts 函数

### 3.2 新增域判别器
- 文件：MFNet/modules/domain_discriminator.py
- 职责：对语义特征进行域二分类。
- 建议新增模块
  - GradientReversalLayer
  - PixelDomainDiscriminator
  - MultiLevelDomainHead

说明：默认先只使用 high-level 特征；MultiLevelDomainHead 作为条件增强接口保留。

### 3.3 新增结构损失
- 文件：MFNet/modules/structure_losses.py
- 职责：封装边界损失与对象一致性损失，兼容 target 伪标签掩码。
- 建议新增模块
  - BoundaryLossWrapper
  - ObjectLossWrapper
  - masked_loss 函数

### 3.4 新增伪标签工具
- 文件：MFNet/modules/pseudo_labeling.py
- 职责：生成 target 高置信伪标签与掩码，并为 prompt 分支预留精炼接口。
- 建议新增模块
  - get_pseudo_label
  - confidence_mask
  - class_balanced_threshold
  - prompt_guided_refine

### 3.5 新增自动提示工具（条件模块）
- 文件：MFNet/modules/prompt_generator.py
- 职责：从 target 高置信区域或响应图中生成点提示、框提示或混合提示。
- 建议新增模块
  - generate_point_prompts
  - generate_box_prompts
  - sample_prompt_set

说明：该文件不是主线必做文件，但若 prompt 分支保留，就必须独立成模块，不能直接把提示逻辑写死在 train 脚本里。

### 3.6 新增配置文件
- 文件：MFNet/configs/uda_struct_v1.yaml
- 职责：集中配置数据路径、损失权重、训练轮数、阈值、日志路径与开关项。
- 建议新增字段
  - lambda_adv
  - lambda_bdy
  - lambda_obj
  - lambda_prompt
  - adv_levels
  - prompt_mode
  - use_prompt_refine

### 3.7 新增数据协议配置
- 文件：MFNet/configs/protocol_isprs.yaml
- 职责：定义主线数据协议与划分。
- 建议字段
  - source_domain
  - target_domain
  - mode
  - train_ids
  - val_ids
  - test_ids

### 3.8 新增实验产物规范配置
- 文件：MFNet/configs/artifact_spec_v1_1.yaml
- 职责：定义每次实验必须导出的表格、图片和统计文件模板。
- 建议字段
  - save_csv
  - save_per_class_iou
  - save_pseudo_stats
  - save_domain_stats
  - save_visual_cases
  - save_failure_cases

## 4. 修改现有文件（必改）

### 4.1 MFNet/UNetFormer_MMSAM.py
- 改造点
  - 在 forward 中新增可选返回特征字典，默认至少支持 high-level 特征输出。
  - 若启用 multi-level 对齐，可返回 mid 与 high 两层特征。
  - 增加 target 侧前向复用主干的能力，保持参数共享。
- 建议接口
  - forward(self, x, y, mode='Train', return_feat=False, feat_levels=('high',))
- 目的
  - 让训练脚本能同时获得分割输出与域对齐所需特征。

### 4.2 MFNet/train.py
- 改造点
  - 保留原脚本用于纯监督训练。
  - 增加提示，说明复杂训练请转到 train_uda_struct_v1.py。
- 目的
  - 不破坏现有基线复现链路。

### 4.3 MFNet/utils.py
- 改造点
  - 增加 source/target 两套 dataloader 的构建接口。
  - 增加 dataloader 同步与无限迭代器函数。
  - 若启用 prompt 分支，可在这里统一放置伪标签统计与样本可视化辅助函数。
- 建议新增函数
  - build_source_target_loaders
  - build_same_domain_loaders
  - build_weak_cross_domain_loaders
  - infinite_loader
  - summarize_pseudo_stats

## 5. 仓库内可复用代码与缺口

### 5.1 可复用部分
- SAM_RS/train.py 可用于参考 BoundaryLoss 与 ObjectLoss 的调用方式。
- MFNet 当前训练脚本与日志格式可作为 same-domain 基线输出模板。

### 5.2 当前仓库缺口
经过当前仓库检查，暂无现成的 GLGAN 或其他 domain adversarial 训练实现可直接复用。因此：
- GRL、域判别器、主干与判别器交替更新逻辑需要新写；
- multi-level 对齐逻辑也需要在本项目中自行实现；
- prompt 分支只能参考文献思路，不能依赖现成仓库模块直接迁移。

这意味着代码改造清单里不应再把不存在的 GLGAN 路径写成“直接迁移”。

## 6. 建议默认损失组合与配置顺序

### 6.1 统一损失形式
- 总损失
  - L_total = L_seg_src + lambda_adv * sum(L_adv_l) + lambda_bdy * L_bdy + lambda_obj * L_obj + lambda_prompt * L_prompt

### 6.2 默认初值
- lambda_adv = 0.001
- lambda_bdy = 0.1
- lambda_obj = 1.0
- lambda_prompt = 0.0 作为默认启动值
- tau = 0.85 作为伪标签初始阈值

### 6.3 启用顺序建议
1. 默认先跑 high-level UDA，adv_levels = [high]。
2. UDA 稳定后，再尝试 adv_levels = [mid, high]。
3. 只有当 fixed-threshold 伪标签已经可用时，才启用 lambda_prompt > 0。

第 6 周起执行补充：
1. 同域先执行结构与模态机制验证（协议 A）。
2. 仅将同域中证据充分的配置迁移到协议 B。
3. 跨域阶段不再做大规模盲目网格，只做门槛通过配置复验。

## 7. 主线训练协议

### 协议 A：Same-domain
- source 与 target 来自同一数据集同一域。
- 启用 L_seg_src + L_bdy + L_obj。
- 若做 prompt feasibility check，可启用 L_prompt，但不建议作为主线默认项。
- 不启用 L_adv。
- 第 6 周起优先执行该协议以完成机制闭环和多 seed 稳态统计。

### 协议 B：Weak-cross-domain
- source 为 Vaihingen，target 为 Potsdam，或反向。
- 启用 L_seg_src + L_adv + L_bdy + L_obj。
- 若 prompt 分支通过阶段门槛，再加 L_prompt。
- 第 6 周之后仅回灌协议 A 中通过门槛的模块与配置。

### 协议 C：Optional stress
- 可选 LoveDA，仅用于压力测试与泛化展示。

## 8. 训练流程改造（按步骤）
1. 读取 source 有标签 batch 与 target 无标签 batch。
2. 前向 source，计算 L_seg_src。
3. 前向 target，生成伪标签与高置信 mask。
4. 若启用 prompt 分支，则基于高置信区域或响应图生成点/框提示，并进行伪标签精炼或一致性约束。
5. 用 source 与 target 的单层或多层特征计算 L_adv。
6. 在 target mask 区域上计算 L_bdy 与 L_obj。
7. 汇总损失更新主干。
8. 再单独更新域判别器。

注：same-domain 协议下跳过对抗训练相关步骤；prompt 步骤仅在配置开启时执行。

## 9. 日志、结果与产物约定

### 9.1 建议目录
- MFNet/runs/uda_struct_v1/
- MFNet/checkpoints/uda_struct_v1/
- MFNet/results/uda_struct_v1/

### 9.2 建议记录项
- 每 iter
  - L_seg
  - L_adv_l
  - L_bdy
  - L_obj
  - L_prompt
  - 伪标签覆盖率
  - 若启用 prompt，还需记录提示数量或提示有效率

- 每 epoch
  - source mIoU
  - target mIoU
  - Boundary F1
  - 域判别器准确率

### 9.3 必须导出的数据
- 配置与环境快照
- loss 与 metric 原始 csv
- 每类 IoU/F1 明细
- 伪标签统计
- 域对齐统计
- best 与 last checkpoint
- 可视化图与 failure cases

## 10. 验收清单

### 10.1 主线验收
- 能完整跑通 1 次 same-domain 训练和 1 次 weak-cross-domain 训练。
- 能保存最优 checkpoint 与可视化预测图。
- 相比 MFNet 纯监督基线，weak-cross-domain target mIoU 有正向提升。
- 能输出至少 1 张组件消融表或对应图像证据。

第 6 周起验收顺序：
1. 先验收 same-domain 多 seed 机制证据。
2. 再验收 weak-cross-domain 的迁移增益与稳定性。

### 10.2 条件增强验收
- multi-level 对齐只有在对比 single-level 后确有收益时才保留。
- prompt 分支只有在相对固定阈值伪标签形成稳定增益或明显稳态改进时才保留。

## 11. 第二版预留接口
- 类别级域对齐。
- 频域一致性或频域分解分支。
- 不确定性感知伪标签重加权。
- prompt embedding 级提示生成，而不仅仅是点/框坐标提示。

这些接口可以保留钩子，但不应在当前版默认实现完毕。
