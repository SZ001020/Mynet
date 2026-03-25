# 遥感语义分割可复现实验框架（通用版，可复用于任意模型）

## 1. 文档目的
本流程用于将任意新模型纳入统一实验体系，保证以下目标同时成立：
- 环境可复现
- 指标可对齐
- 结果可追踪
- 脚本可复用

本流程不依赖具体模型结构，可用于 CNN、Transformer、Mamba、SAM 微调等任意分割模型。

## 2. 通用框架总览
统一实验框架分为四层：
- 层 A：环境层（数据、依赖、GPU、随机性）
- 层 B：训练层（统一超参数入口、统一日志协议）
- 层 C：评估层（统一指标计算器、统一数据切分）
- 层 D：汇总层（统一报表、统一曲线、统一对比表）

建议所有模型都遵循同一目录与命名规范，避免后期横向比较困难。

## 3. 通用目录规范
建议约定以下目录（可按项目实际调整）：
- runs
  - experiment_name
    - logs
    - checkpoints
    - reports
    - figures

推荐说明：
- logs：保存训练和验证 CSV
- checkpoints：保存模型权重
- reports：保存 md 和表格结果
- figures：保存曲线图、可视化图

## 4. 统一环境变量协议
所有模型统一读取以下环境变量：
- EXP_DATA_ROOT：数据集根目录
- EXP_DATASET：数据集名称
- EXP_SPLIT：数据切分标识（例如 fold1）
- EXP_SEED：随机种子
- EXP_LOG_DIR：日志输出目录
- EXP_CKPT_DIR：权重输出目录
- EXP_REPORT_DIR：汇总输出目录
- EXP_BATCH_SIZE：批大小
- EXP_WINDOW_SIZE：裁块大小
- EXP_BASE_LR：学习率
- EXP_EVAL_STRIDE：推理步长
- EXP_NUM_WORKERS：数据线程数
- EXP_DEVICE：设备标识（例如 cuda:0）

可选模型专用变量示例：
- EXP_PRETRAIN_CKPT：预训练权重路径
- EXP_BACKBONE：主干类型
- EXP_PROMPT_MODE：参数高效微调模式

性能相关通用变量建议：
- EXP_PERF_MODE：是否开启高性能模式（例如 TF32、benchmark）
- EXP_NUM_WORKERS：DataLoader worker 数
- EXP_PIN_MEMORY：是否开启 pin_memory
- EXP_PERSISTENT_WORKERS：是否开启 persistent_workers
- EXP_PREFETCH_FACTOR：DataLoader 预取因子
- EXP_USE_AMP：是否开启自动混合精度
- EXP_MICRO_BS：微批次大小（用于梯度累积或分块反传）
- OMP_NUM_THREADS：OpenMP 线程数
- MKL_NUM_THREADS：MKL 线程数

注意：OMP_NUM_THREADS 和 MKL_NUM_THREADS 不应为 0。

## 5. 环境配置检查标准
每次实验前执行一次就绪检查，输出 PASS 或 FAIL 清单。

### 5.1 必查项
- 数据根目录存在
- 训练集、验证集、测试集样本可读
- 标签映射文件存在且可读
- 预训练权重存在（如模型依赖）
- 训练脚本可导入
- CUDA 可用（若使用 GPU）
- 关键依赖可导入（torch、numpy、skimage、opencv、timm 等）

建议新增检查项：
- 输出目录存在（logs/checkpoints/reports）
- 训练脚本中模型导入路径可用
- 默认预训练权重路径可访问

### 5.2 推荐检查输出格式
每行输出：
- 状态
- 检查项名称
- 详细路径或异常信息

示例：
- PASS | dataset root | /path/to/dataset
- FAIL | pretrain ckpt | /path/to/missing_ckpt

## 6. 训练脚本最小接入规范
任意新模型训练脚本需满足以下最小要求。

### 6.1 固定随机种子
统一固定以下随机源：
- Python random
- numpy
- torch
- torch cuda
并设置：
- cudnn.deterministic = True
- cudnn.benchmark = False

实践建议：
- 复现实验阶段使用 deterministic=True。
- 追求吞吐阶段可切换 benchmark=True（通常会更快）。

### 6.2 统一日志字段
每个验证周期写一行 CSV，字段固定为：
- model
- dataset
- split
- seed
- epoch
- iter
- train_loss
- val_metric_main
- val_metric_aux
- best_metric_main
- lr
- batch_size
- window_size
- eval_stride
- ckpt_path
- timestamp

说明：
- val_metric_main 建议为 mIoU
- val_metric_aux 可放 mean F1 或 Boundary F1

建议扩展字段（用于统一横向分析）：
- total_acc
- mean_f1
- kappa
- mean_miou
- roads_f1
- buildings_f1
- low_veg_f1
- trees_f1
- cars_f1
- clutter_f1

### 6.3 权重保存规则
建议按统一命名保存：
- model_dataset_split_seed_epoch_metric.pth

必须保证：
- 每个实验目录至少保留 best 与 last 两个权重
- 日志中的 ckpt_path 可追溯到真实文件

建议额外记录：
- 训练是否开启 AMP
- DataLoader workers/prefetch 参数
- OMP/MKL 线程设置

## 7. 统一指标协议（核心）

### 7.1 指标集合
建议统一计算以下指标：
- 主指标：mIoU
- 辅指标：mean F1
- 边界指标：Boundary F1
- 可选：OA、Kappa、per-class IoU

### 7.2 强制同口径原则
要保证模型间可比，必须统一：
- 测试切分
- 标签映射
- 忽略类策略
- 后处理策略
- 阈值策略（若含概率阈值）

实践补充：
- 训练脚本内置评估口径不一致时，必须统一离线评估器复评。

### 7.3 统一评估器接口（建议）
建议设计一个独立评估器，输入为：
- 预测图目录
- 真值目录
- 标签定义
输出为：
- metrics.json
- per_class_metrics.csv

这样可避免不同训练脚本内置评估实现差异带来的口径偏差。

## 8. 统一结果汇总流程

### 8.1 汇总输入
汇总程序统一读取：
- 每次实验的 CSV 日志
- 最终 metrics.json
- 可选训练曲线缓存

### 8.2 汇总输出
每次批量实验后自动生成：
- baseline_summary.md
- baseline_table.csv
- val_curves.png
- loss_curves.png
- rank_table.csv

建议追加：
- detailed_metrics_table.csv（统一保存 OA/F1/Kappa/每类 F1）

### 8.3 汇总最小字段
建议汇总表至少包含：
- model
- dataset
- split
- seed
- best_epoch
- best_mIoU
- mean_F1
- boundary_F1
- params
- trainable_params
- flops
- ckpt_path

## 9. 新模型接入操作清单（可直接复用）

1. 复制一个已有训练脚本作为模板。
2. 替换模型构建与前向逻辑。
3. 保留统一环境变量读取逻辑。
4. 保留统一随机种子逻辑。
5. 保留统一 CSV 写入逻辑。
6. 接入统一评估器输出 mIoU、mean F1、Boundary F1。
7. 接入统一 checkpoint 命名和保存目录。
8. 将日志输出接到统一汇总脚本。
9. 跑 1 到 2 epoch 的 smoke test。
10. 通过后再跑完整训练。

## 10. 复现实验推荐顺序
- 步骤 1：执行环境检查并保存检查报告。
- 步骤 2：运行 smoke test（快速验证管线通畅）。
- 步骤 3：运行正式训练。
- 步骤 4：统一评估器离线评测。
- 步骤 5：生成统一汇总表与曲线图。
- 步骤 6：多随机种子复跑并报告均值与方差。

### 10.1 调试模式与正式模式
调试模式（快速验证管线）：
- 降低每 epoch 步数
- 减少验证样本数量
- 缩短总 epoch

正式模式（最终报告）：
- 恢复完整步数与完整验证集
- 固定验证频率
- 多随机种子复跑

## 11. 常见失败模式与排查

### 11.1 日志有写入但无法汇总
- 原因：字段名不一致或字段缺失
- 处理：严格使用统一字段模板，不允许自定义字段替换主字段

### 11.2 指标看起来异常高或异常低
- 原因：测试集不一致、标签映射不一致、忽略类不一致
- 处理：统一评估器重新离线计算并核对标签映射

### 11.3 同一配置多次结果差异过大
- 原因：随机性未完全固定、数据加载顺序未固定
- 处理：检查种子设置、DataLoader 随机源、cudnn 选项

### 11.4 权重存在但无法恢复
- 原因：模型定义变更导致 key 不匹配
- 处理：记录模型版本和配置快照，确保训练与推理版本一致

### 11.5 Scheduler 调用顺序告警
- 原因：在 optimizer.step() 前调用 scheduler.step()
- 处理：将 scheduler.step() 放在 epoch 末尾，确保在 optimizer.step() 之后。

### 11.6 AMP/接口弃用告警
- 原因：使用旧版 torch.cuda.amp 接口
- 处理：迁移到 torch.amp 统一接口。

### 11.7 旧推理接口告警
- 原因：使用 volatile=True 或过时进度条接口
- 处理：
- 推理阶段使用 with torch.no_grad()。
- 进度条使用 tqdm.auto.tqdm。

### 11.8 DataLoader 与线程配置不当
- 现象：GPU 利用率低、CPU 空转或初始化告警
- 处理：
- 合理提升 EXP_NUM_WORKERS（例如 8 到 24 区间试验）。
- 开启 pin_memory 和 persistent_workers。
- 避免 OMP/MKL 线程设置为 0。

### 11.9 OOM 通用处理
- 开启 AMP。
- 使用微批次分块反传。
- 降低 batch size 或输入分辨率。

### 11.10 自动调参评估口径（新增）
- 现象：在固定 epoch_steps 的短跑基准里，batch_size 增大后 elapsed_sec 常常也会增大。
- 原因：固定步数时，总样本数随 batch_size 线性增加，仅比较总耗时会误判“更慢”。
- 处理原则：自动调参应以吞吐（samples/s）作为主排序指标，耗时作为辅指标。

推荐同时给出两类结论：
- wall-time best：最短耗时组合
- throughput best：最高吞吐组合

推荐计算：
- samples = epochs * epoch_steps * batch_size
- throughput = samples / elapsed_sec

落地建议：
- 默认采用 throughput best 作为正式训练参数起点。
- 当两组吞吐接近时，优先选择显存更安全、训练更稳定的组合作为生产配置。

### 11.11 验证触发节奏不一致（新增）
- 现象：有的脚本在 epoch 中间按 step 做验证，日志上看起来像“半个 epoch 就评估”。
- 处理：统一在日志中记录 `epoch + iter`，并在配置中明确验证触发规则（按 step 或按 epoch）。

### 11.12 多卡设备绑定被硬编码覆盖（新增）
- 现象：设置了 CUDA_VISIBLE_DEVICES，但实际仍只占用默认 GPU。
- 处理：禁止硬编码 `cuda:0`；优先通过环境变量传递设备；并发训练前后都用 nvidia-smi 复核。

### 11.13 输入尺寸与模型结构约束冲突（新增）
- 现象：切换 patch/window 尺寸后报 shape mismatch。
- 处理：把输入尺寸纳入实验配置快照；每次改尺寸先做 smoke test，再启动正式训练。

### 11.14 汇总缺失详细指标或曲线（新增）
- 现象：报告中出现 N/A 或 No curve data found。
- 处理：检查 CSV 字段是否完整、是否至少有一条验证记录；必要时从 train.log 回填 detailed metrics。

## 12. 通用验收标准
一个模型接入框架后，满足以下条件视为通过：
- 环境检查无 FAIL
- smoke test 可完整跑通
- 正式训练能产生日志、权重、评估结果
- 汇总脚本能识别该模型并写入总表
- 指标可在统一评估器上复算一致

## 13. 建议扩展（后续）
- 增加配置文件驱动训练（yaml）
- 增加实验元数据记录（git commit、配置哈希、设备信息）
- 增加结果数据库（便于长期横向对比）
- 增加自动告警（训练崩溃、指标回退）
- 增加自动吞吐调优脚本（自动扫描 workers、micro_bs、batch_size 组合）

## 14. 最小执行模板
建议形成三条固定命令：
- 命令 A：环境就绪检查
- 命令 B：单模型训练
- 命令 C：统一汇总

只要新模型能满足这三条命令可执行，即可纳入同一可复现实验体系。
