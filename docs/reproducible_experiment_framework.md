# SSRS 可复现实验框架搭建流程（通用于后续模型）

## 1. 目标
本流程用于把不同模型纳入同一套可复现实验管线，核心包含三部分：
- 环境配置检查
- 统一指标记录
- 统一结果汇总

适用对象：MFNet、FTransUNet、ASMFNet，以及后续新增模型。

## 2. 基础改造
仓库目标能力：
- 统一数据根路径与数据集选择（环境变量驱动）
- 统一随机种子入口
- 统一训练日志 CSV 字段（基础字段 + 详细指标字段）
- 统一汇总脚本（自动生成基线表和曲线图）
- 一键运行脚本（串行跑三模型并汇总）

## 3. 环境配置规范

### 3.1 统一环境变量
建议在每次实验前设置：

```bash
export SSRS_DATA_ROOT=/root/SSRS/autodl-tmp/dataset
export SSRS_DATASET=Vaihingen
export SSRS_SEED=42
export SSRS_LOG_DIR=/root/SSRS/runs/week1_baseline
export SSRS_BATCH_SIZE=10
export SSRS_WINDOW_SIZE=256
export SSRS_BASE_LR=0.01
export SSRS_EVAL_STRIDE=32
export SSRS_PERF_MODE=1
export SSRS_NUM_WORKERS=16
export SSRS_PIN_MEMORY=1
export SSRS_PERSISTENT_WORKERS=1
export SSRS_PREFETCH_FACTOR=4
export SSRS_MFNET_USE_AMP=1
export SSRS_MFNET_MICRO_BS=2
export SSRS_USE_STRUCTURE_LOSS=1
export SSRS_LOSS_MODE=SEG+BDY+OBJ
export SSRS_LAMBDA_BDY=0.1
export SSRS_LAMBDA_OBJ=1.0
export SSRS_EPOCH_STEPS=1000
export SSRS_EVAL_NUM_TILES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

说明：
- ASMFNet 默认从 SSRS_DATA_ROOT + Vaihingen 读取数据。
- 若 ASMFNet 数据目录需要单独指定，可设置 ASMFNET_DATA_ROOT。
- MFNet 的 SAM 权重支持环境变量：
  - SSRS_MFNET_SAM_ENCODER（默认 vit_l）
  - SSRS_MFNET_SAM_CKPT（默认 weights/sam_vit_l_0b3195.pth）
- 建议不要把 OMP_NUM_THREADS 或 MKL_NUM_THREADS 设为 0，0 会触发 libgomp 警告并影响性能。

### 3.3 Debug 快速检查（新增）
建议训练启动前做 4 项快速检查：
- 导入检查：训练脚本中的模型导入路径是否有效。
- 权重检查：默认 checkpoint 是否存在，是否需要 fallback 路径。
- 输出目录检查：权重保存目录是否提前创建。
- 日志检查：CSV 是否已创建并包含表头。

常见现象解释：
- 只有 CSV 表头没有数据行：通常是训练在第一个验证点前中断，或还没跑到验证周期。
- 汇总图显示 No curve data found：通常是日志里没有任何 val_metric 数据行。

### 3.2 一次性就绪检查（建议每次开跑前执行）
检查项清单：
- 数据根目录存在
- 样本数据可读（影像/DSM/标签/eroded 标签）
- 训练脚本可导入
- 预训练权重存在
- GPU 可用

建议脚本化输出 PASS/FAIL 清单，避免训练开始后才报错。

## 4. 统一指标记录规范

### 4.1 统一 CSV 字段
所有模型训练日志至少写入以下基础字段：
- model
- dataset
- seed
- epoch
- iter
- train_loss
- val_metric
- best_val_metric
- lr
- batch_size
- window_size
- stride

建议同时写入以下扩展字段（近期已在三模型统一）：
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
- timestamp

### 4.2 当前实现说明
当前三个模型均已写入统一字段，但 val_metric 来自各模型现有评估函数输出。
- 如果要严格同口径比较 mIoU/F1/Boundary-F1，需在同一 evaluator 上对保存预测图复评。

### 4.3 推荐的同口径指标策略
- 主指标：mIoU
- 辅指标：mean F1
- 边界指标：Boundary F1
- 所有指标都基于同一测试集切分和同一标签映射计算。

## 5. 统一结果汇总流程

### 5.1 自动汇总脚本
使用脚本：
- utils/summarize_week1_baseline.py

功能：
- 读取 SSRS_LOG_DIR 下所有 CSV
- 生成基线汇总表：week1_baseline_summary.md
- 生成验证曲线图：week1_val_curves.png
- 生成训练损失曲线图：week1_loss_curves.png
- 从 train.log 抽取详细指标并写入汇总表（Total Acc/mean F1/Kappa/mean MIoU/各类 F1）

### 5.2 一键运行脚本
使用脚本：
- utils/run_week1_baselines.sh

功能：
1. 统一导出环境变量（可覆盖）
2. 串行运行 MFNet / FTransUNet / ASMFNet
3. 自动调用汇总脚本输出报告

### 5.3 第 2 周同域增强一键脚本（新增）
使用脚本：
- utils/run_week2_same_domain.sh

功能：
1. 执行训练前就绪检查（数据/GPU/脚本/关键样本）
2. 运行基线组（MFNet + SEG）
3. 运行结构约束组（MFNet + SEG+BDY+OBJ）
4. 自动输出 week2_same_domain_summary.md（含两组对比和增益）

输出目录（默认）：
- runs/week2_same_domain/<timestamp>/

## 6. 文件落盘位置

### 6.1 日志与汇总
默认输出目录：
- /root/SSRS/runs/week1_baseline

包含：
- 各模型 CSV
- week1_baseline_summary.md
- week1_val_curves.png
- week1_loss_curves.png

### 6.2 权重输出
- MFNet
  - /root/SSRS/MFNet/resultsv
  - /root/SSRS/MFNet/resultsp
  - /root/SSRS/MFNet/resultsh
- FTransUNet
  - /root/SSRS/FTransUNet/results_final
- ASMFNet
  - /root/SSRS/ASMFNet/res2

## 7. 迁移到“其他新模型”的复用模板
将新模型纳入本框架时，按以下最小步骤改造：

1. 接入统一环境变量
- SSRS_DATA_ROOT
- SSRS_DATASET
- SSRS_SEED
- SSRS_LOG_DIR
- SSRS_BASE_LR
- SSRS_BATCH_SIZE
- SSRS_WINDOW_SIZE
- SSRS_EVAL_STRIDE

2. 接入统一随机种子
- random / numpy / torch / cuda 全部固定
- cudnn.deterministic=True, cudnn.benchmark=False

3. 接入统一 CSV 记录
- 按 12 个标准字段写日志
- 每次验证后写一行

4. 保持权重输出目录稳定
- 统一保存在模型目录下固定子目录
- 文件名包含 epoch 与 val_metric

5. 接入统一汇总
- 确保 CSV 落在 SSRS_LOG_DIR
- 直接复用 summarize_week1_baseline.py

## 8. 复现实验建议执行顺序
1. 运行环境就绪检查
2. 跑单模型 smoke test（1-2 epoch）
3. 跑完整训练
4. 自动汇总并检查异常点
5. 复跑不同 seed（至少 3 个）
6. 导出最终基线表

### 8.1 调试模式与正式模式（新增）
调试模式（快速看趋势）：
- SSRS_EPOCH_STEPS=200
- SSRS_EPOCHS=10
- SSRS_SAVE_EPOCH=2
- SSRS_EVAL_NUM_TILES=1

正式模式（完整结果）：
- SSRS_EPOCH_STEPS=1000
- SSRS_EPOCHS=50
- SSRS_SAVE_EPOCH=1
- SSRS_EVAL_NUM_TILES=0

## 9. 常见问题与处理

### 9.1 权重文件缺失
- 现象：训练启动即报 checkpoint not found
- 处理：
  - 设置模型专用权重路径环境变量
  - 或在模型代码中增加本地 fallback 权重逻辑

### 9.4 Scheduler 警告
- 现象：Detected call of lr_scheduler.step() before optimizer.step()
- 处理：将 scheduler.step() 放到 epoch 末尾，在 optimizer.step() 之后调用。

### 9.5 AMP 与旧 API 警告
- 现象：torch.cuda.amp.* deprecated
- 处理：改用 torch.amp.GradScaler('cuda', ...) 和 torch.amp.autocast('cuda', ...)。

### 9.6 volatile 与 tqdm 旧接口警告
- 现象：volatile was removed、tqdm_notebook deprecated
- 处理：
- 推理张量直接使用 torch.from_numpy(...).cuda()，并配合 with torch.no_grad()。
- tqdm 统一改用 tqdm.auto.tqdm。

### 9.7 交叉熵 reduction 警告
- 现象：size_average and reduce args will be deprecated
- 处理：将 F.cross_entropy 的旧参数改为 reduction='mean' 或 reduction='sum'。

### 9.8 Mean of empty slice 警告
- 现象：numpy Mean of empty slice
- 处理：移动平均窗口包含当前 step，避免切片为空。

### 9.9 OOM 处理经验（新增）
- 首选：开启 AMP（SSRS_MFNET_USE_AMP=1）。
- 次选：开启微批次（SSRS_MFNET_MICRO_BS=2 或 3）。
- 再次选：降低 SSRS_BATCH_SIZE 或 SSRS_WINDOW_SIZE。

### 9.10 自动调参口径（新增）
- 现象：固定 epoch_steps 的短跑中，batch_size 越大时总耗时可能变长。
- 原因：固定步数下，batch_size 越大，总处理样本数越多，因此只看 elapsed_sec 会产生误判。
- 正确口径：优先比较吞吐（samples/s），总耗时仅作为次级参考。
- 建议同时输出两套最优：
- 最短耗时最优（wall time best）
- 最高吞吐最优（throughput best）

推荐计算方式：
- samples = epochs * epoch_steps * batch_size
- throughput = samples / elapsed_sec

实践建议：
- 以吞吐最优参数作为默认训练参数。
- 若两组吞吐接近，优先选择更稳的组合（OOM 风险更低、波动更小）。

### 9.11 验证节奏导致“半个 epoch 就评估”误解（新增）
- 现象：日志显示 epoch 尚未结束就触发验证，容易误判为训练流程异常。
- 原因：代码按 step 间隔触发验证，而不是仅在 epoch 末尾验证。
- 处理：在日志中同时记录 epoch 与 iter，并在文档中明确“验证触发条件（按步数或按 epoch）”。

### 9.12 多卡并行时 GPU 未按预期工作（新增）
- 现象：期望用 GPU1 训练，但进程始终占用 GPU0 或只用单卡。
- 原因：脚本中存在硬编码 device，覆盖了外部 CUDA_VISIBLE_DEVICES。
- 处理：统一以环境变量决定设备；移除硬编码 `cuda:0`；并发任务前先用 nvidia-smi 二次确认绑定。

### 9.13 ASMFNet 输入尺寸与模块约束冲突（新增）
- 现象：训练时报尺寸不匹配，常见于 224 与 256 切换时。
- 原因：部分分支或预训练结构对输入尺寸有隐式约束。
- 处理：在实验配置里固定窗口尺寸；切换尺寸时先做 1 到 2 epoch smoke test 再正式开跑。

### 9.14 汇总脚本出现 N/A 或空曲线（新增）
- 现象：summary 表中 detailed metrics 为 N/A，或图中显示 No curve data found。
- 原因：日志缺字段、未到验证点、或 train.log 未写入可解析指标行。
- 处理：先检查 CSV 表头与首行数据，再检查 train.log 是否包含指标关键字；必要时补跑短程验证生成最小可汇总样本。

### 9.2 日志格式不一致
- 现象：汇总脚本读取失败或字段为空
- 处理：
  - 检查 CSV 字段名是否与标准字段一致
  - 保证字段顺序和命名固定

### 9.3 指标口径不一致
- 现象：不同模型 mIoU 不可比
- 处理：
  - 使用同一 evaluator 对预测图离线重评
  - 严格固定测试集、标签映射、忽略类策略

## 10. 最小可执行命令

```bash
cd /root/SSRS
export SSRS_DATA_ROOT=/root/SSRS/autodl-tmp/dataset
export SSRS_DATASET=Vaihingen
export SSRS_SEED=42
export SSRS_LOG_DIR=/root/SSRS/runs/week1_baseline
export SSRS_BATCH_SIZE=10
export SSRS_WINDOW_SIZE=256
export SSRS_BASE_LR=0.01
export SSRS_EVAL_STRIDE=32
export SSRS_PERF_MODE=1
export SSRS_NUM_WORKERS=16
export SSRS_PIN_MEMORY=1
export SSRS_PERSISTENT_WORKERS=1
export SSRS_PREFETCH_FACTOR=4
export SSRS_MFNET_USE_AMP=1
export SSRS_MFNET_MICRO_BS=2
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

/root/SSRS/utils/run_week1_baselines.sh
```

第 2 周同域增强最小命令：

```bash
cd /root/Mynet
export SSRS_DATA_ROOT=/root/Mynet/autodl-tmp/dataset
export SSRS_DATASET=Vaihingen
export SSRS_SEED=42
export SSRS_EPOCH_STEPS=1000
export SSRS_EPOCHS=50
export SSRS_LAMBDA_BDY=0.1
export SSRS_LAMBDA_OBJ=1.0

./utils/run_week2_same_domain.sh
```

执行完成后查看：
- runs/week2_same_domain/<timestamp>/week2_same_domain_summary.md

执行完成后直接查看：
- /root/SSRS/runs/week1_baseline/week1_baseline_summary.md
- /root/SSRS/runs/week1_baseline/week1_val_curves.png
- /root/SSRS/runs/week1_baseline/week1_loss_curves.png
