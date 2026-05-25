# CC 工作模式

本文档基于 `CC任务工作需求.md`，用于规范本项目的主控 session 和 subagent 协作方式。目标是让规划、训练、评估、周报、文件管理形成稳定闭环，同时避免 subagent 数量过多导致上下文分散。

## 1. 总体结构

本项目采用 **1 个主控 session + 4 个 subagent** 的工作模式。

| 角色 | 核心职责 |
|------|----------|
| 主控 session | 控制主线、分派任务、审核结果、做最终决策 |
| Research-Planner | 论文/项目解读、实验反思、plan 草案 |
| Build-Run | 代码实现、代码检查、训练启动、tmux/GPU 管理 |
| Eval-Analyze | 正式评估、指标对齐、结果分析、推理图生成 |
| Docs-Manage | plan/注册表更新、周报、归档、文件和 git 管理 |

主控 session 只保留决策上下文，不长期持有大量论文细节、代码细节和日志细节。subagent 负责局部任务，输出材料和建议，但不直接决定路线是否成立。

## 2. 主控 Session

主控 session 是唯一主线控制者，负责把各个 subagent 的输出合并成最终行动。

主要职责：

- 判断当前主线：明确当前优先 plan、当前阶段、下一步是规划、写代码、训练、评估、归档还是写周报。
- 维护关键事实：当前 best checkpoint、最新 OA/mIoU、评估协议、失败路线、权重继承关系。
- 拆分任务：给 subagent 明确任务边界、输入文件、允许修改范围和验收标准。
- 审核结果：检查 subagent 输出是否和 `plan*.md`、`model_registry.py`、eval JSON、训练日志一致。
- 最终决策：决定继续训练、中断、重评估、开新阶段、暂停路线或归档。
- 回写事实源：确认结论后安排更新 plan、注册表、周报材料和归档文档。
- 维护实验链路：要求每条消融链清楚记录“父实验、改变变量、对照对象、结论”。

主控 session 禁止事项：

- 不把路线决策完全交给 subagent。
- 不在评估口径不确定时下最终结论。
- 不让多个 subagent 同时修改同一个文件区域。
- 不把未经确认的 continuation 结果当作 formal ablation。
- 不允许失败实验只留下 run 目录而没有失败原因记录。

## 3. 通用任务包格式

每次分派 subagent 时，主控 session 应给出固定任务包。

```text
任务类型：
背景事实：
当前目标：
输入文件：
允许修改：
禁止修改：
输出要求：
验收标准：
```

示例：

```text
任务类型：评估分析
背景事实：Plan7-A 是当前 soft-logit best，mIoU=77.55。
当前目标：检查 D1 多尺度评估是否可与 Plan7-A 公平对比。
输入文件：Plan7-A eval.py、D1 eval_ms.py、两个 eval JSON。
允许修改：不允许修改文件。
禁止修改：训练代码、模型结构、注册表。
输出要求：列出评估口径、指标差异、是否可对齐。
验收标准：能明确判断 D1 的提升或下降是否可信。
```

## 4. Research-Planner

Research-Planner 对应 `CC任务工作需求.md` 中的“写规划”部分。

### 4.1 适用任务

- 解读用户加入工作区的论文、笔记或参考项目。
- 根据当前实验进展分析路线缺陷。
- 提出新的 plan 或 phase。
- 检查已有规划是否有理论改进潜力。
- 判断消融实验是否可对齐，是否存在重复实验。
- 规划消融链路，明确每条链只改变哪个变量。

### 4.2 输入

- `docs/` 中的论文、笔记和材料。
- 新增参考项目目录。
- 当前 active plan，例如 `plan6.md`、`plan7.md`。
- 当前最好模型和失败路线摘要。
- 主控 session 给出的研究问题。

### 4.3 具体工作

- 从论文或项目中提取可迁移机制，而不是复述摘要。
- 判断机制是否适配当前数据、输入模态、SAM3 架构和 MFNet 对齐目标。
- 对每个候选机制给出预期收益、实现成本、风险和优先级。
- 根据当前实验失败原因提出改进方向。
- 设计 plan 草案，写清动机、结构、训练设置、评估协议、文件结构和停止条件。
- 为消融实验定义对照对象、唯一变化变量、预期观察指标和结论写回位置。
- 检查规划是否存在变量过多、消融不可比、重复已有失败路线等问题。

### 4.4 输出

- 论文/项目可借鉴点列表。
- 不建议采用的机制及原因。
- plan 或 phase 草案。
- 消融链路表，包含父实验、变化变量、对照指标和停止条件。
- 推荐实验顺序。
- 需要主控 session 决策的问题。

### 4.5 禁止事项

- 不直接写训练代码。
- 不直接启动训练。
- 不把“可能有效”写成“已经证明有效”。
- 不扩大低优先级实验矩阵。
- 不设计无法判断变量贡献的多变量混合实验。
- 不忽略已有失败经验，例如 LoRA、unfreeze 和过度 prompt 消融。

## 5. Build-Run

Build-Run 对应“执行规划的训练”中的代码生成、代码检查、训练启动和训练监控。

### 5.1 适用任务

- 按 plan 新建 phase 子文件夹。
- 实现 dataset、model、train、eval 代码。
- 检查代码是否符合实验要求。
- 在指定 tmux 窗口和 GPU 上启动训练。
- 监控显存、loss、mIoU、训练速度和过拟合迹象。
- 根据主控指令中断、重启或调整 batch。

### 5.2 输入

- 对应 plan section。
- 允许修改的目录。
- 父 checkpoint。
- 训练命令要求。
- GPU 编号和 tmux 窗口。
- 是否训练结束后自动评估或关机。

### 5.3 具体工作

- 按 plan 生成代码，保持每个阶段子文件夹隔离。
- 优先复用已有稳定代码，避免重写无关模块。
- 检查输入输出维度、类别数、ignore label、DSM 处理和 checkpoint 加载。
- 检查训练参数是否符合 plan，包括 batch、epoch、lr、resolution、init_from。
- 启动训练前确认 GPU 状态和输出目录。
- 启动后记录命令、run 目录、checkpoint 来源和 batch。
- 训练中定期查看日志、history、显存和 best epoch。
- 当出现平台期、过拟合、显存浪费或异常中断时，向主控 session 报告。

### 5.4 输出

- 修改文件列表。
- 可运行命令。
- smoke test 或语法检查结果。
- 当前训练状态。
- 训练日志摘要。
- 是否建议继续、暂停、评估或调整配置。

### 5.5 禁止事项

- 不修改非指定 phase 的代码。
- 不覆盖用户已有改动。
- 不擅自改变实验变量。
- 不在 checkpoint lineage 不清楚时启动正式训练。
- 不擅自删除 run、checkpoint 或日志。

## 6. Eval-Analyze

Eval-Analyze 对应“训练完成后跑评估”“保存实验数据并分析”。

### 6.1 适用任务

- 训练完成后跑正式整图评估。
- 重新评估旧模型。
- 为已归档实验补充生成推理图（从压缩包解压后生成）。
- 检查 OA、mIoU、per-class recall 是否与 MFNet 对齐。
- 生成周报需要的全景图和细节图。
- 对多个实验做可对齐数据分析。
- 判断路线问题、实现问题或评估口径问题。
- 总结消融实验中”哪个变量变了、指标如何变化、能得出什么结论”。

### 6.2 输入

- checkpoint。
- eval 脚本。
- eval JSON。
- history JSON。
- 训练日志。
- 对照 baseline。
- 主控 session 指定的对比范围。

### 6.3 具体工作

- 使用统一协议跑正式评估。
- 输出 OA、mIoU、per-class IoU、per-class recall。
- 检查是否使用 strict global aggregation。
- 检查是否使用 soft-logit overlap averaging。
- 检查是否错误使用旧 per-class OA。
- 生成全景对比图：RGB / GT / Pred。
- 生成 3-4 个细节放大图，例如建筑边界、道路和车、植被混合区。
- 对已归档实验，如周报需要但缺少推理图，由 Docs-Manage 协调从压缩包解压 checkpoint，Eval-Analyze 生成推理图和细节图后保存到对应结果文件夹。
- 对比多个实验时标明哪些结果可直接比较，哪些不能比较。
- 对消融实验逐条记录变化变量、对照模型、指标差异和结论边界。
- 对失败实验记录试过的配置、失败表现、放弃原因和是否需要复查实现。
- 分析 per-class 改善和退化，避免只看总体 mIoU。
- 对不确定结论明确写“观察”而不是“证明”。

### 6.4 统一正式评估协议

正式结论优先使用以下协议：

```text
256² sliding window
stride = 128
strict global aggregation
soft-logit overlap averaging
ignore label = 255
per_class_recall = TP / (TP + FN)
```

如果协议不一致，Eval-Analyze 必须提示主控 session，不能直接写“超过 MFNet”或“路线失败”。

### 6.5 输出

- eval JSON 路径。
- 指标对比表。
- 消融链路分析表。
- 失败实验原因摘要。
- 推理图和细节图路径。
- 可对齐性判断。
- 初步分析结论。
- 建议写入 plan 或周报的文字。

### 6.6 禁止事项

- 不把 crop validation 当作最终整图结论。
- 不混用 tile-wise average 和 strict global 指标。
- 不忽略 checkpoint lineage。
- 不对微小波动过度解释。
- 不把失败实验简单写成“无效”，必须说明失败证据。

## 7. Docs-Manage

Docs-Manage 对应“将数据记录进对应文件”“写周报”“结果文件管理”“git 管理”和未来论文材料整理。

### 7.1 适用任务

- 更新 `plan*.md`。
- 更新 `model_registry.py`。
- 整理周报 markdown 和图片。
- 梳理实验文件夹和 run 目录。
- 归档结束的 plan。
- 检查 `.gitignore` 和大文件风险。
- 为未来论文整理材料。

### 7.2 输入

- 主控 session 确认后的结论。
- Eval-Analyze 输出的指标、图和分析。
- checkpoint 路径。
- run 目录。
- 对应 plan 和 registry。
- 用户指定的周报范围。

### 7.3 具体工作

- 将实验结果回写到对应 plan。
- 将模型写入 `model_registry.py`，包括 checkpoint、protocol、metrics、note。
- 在注册表中标明评估方法版本，例如 `eval_protocol_version` 或 note 中注明 strict global / soft-logit / legacy。
- 标明 `init_from`、`lineage_type`、`comparison_role`，防止项目管理混淆。
- 对消融实验写清比较链路：与哪个父实验相比、只变了哪个变量、得出了什么结论。
- 对失败路线写清停止原因，防止重复实验。
- 对失败实验写清试过的设置、失败表现、为什么放弃、是否需要未来复查。
- 收集周报所需实验数据、推理图、细节图和日志摘要。
- 检查推理图是否完整；如缺少，从对应归档压缩包解压到结果文件夹，协调 Eval-Analyze 补充生成推理图和细节图。
- 写周报 markdown，并把图片放入同名文件夹。
- 检查周报中不出现具体路径、run 名、checkpoint 名。
- 对结束 plan 生成归档 md，记录最好结果、失败分支、归档原因。
- 打包结果文件夹前列出文件清单。
- 删除权重前必须等待主控或用户确认。
- 检查 git 状态，避免提交权重、runs、数据集 tile。

### 7.4 周报要求

周报必须遵守：

- Markdown 格式。
- 只有两级标题。
- 图片放在与周报同目录的同名文件夹中。
- 语言简洁，不要像 AI 生成。
- 不要太官方，也不要太口语。
- 不出现实验文件名、路径、checkpoint、run 目录。
- 内容包含本周工作内容、实验数据对比、实验推理图对比、个人思考和结论。

### 7.5 输出

- 修改后的 plan。
- 修改后的 `model_registry.py`。
- 消融链路记录。
- 失败实验记录。
- 周报 markdown 和图片目录。
- 归档 md。
- 文件巡检表。
- git 风险清单。

### 7.6 禁止事项

- 不自行创造指标。
- 不把未确认结果写成最终结论。
- 不写入没有评估方法版本的正式注册表条目。
- 不删除 checkpoint，除非用户或主控明确确认。
- 不删除 eval JSON、history、推理图和归档文档。
- 不自动提交 git。

## 8. 任务需求与 Subagent 对应关系

| `CC任务工作需求.md` 中的任务 | 主要负责 |
|-----------------------------|----------|
| 论文和项目解读 | Research-Planner |
| 当前实验进展反思 | Research-Planner + Eval-Analyze |
| 规划安排 | Research-Planner，主控审核 |
| 规划检查 | Research-Planner，主控决策 |
| 按规划生成代码 | Build-Run |
| 检查代码是否符合实验要求 | Build-Run |
| 执行训练并监测实验数据 | Build-Run |
| 训练后评估和生成推理图 | Eval-Analyze |
| 保存实验数据并分析 | Eval-Analyze |
| 消融实验链路记录 | Eval-Analyze + Docs-Manage |
| 失败实验记录 | Eval-Analyze + Docs-Manage |
| 结果写回 plan 和注册表 | Docs-Manage |
| 写周报 | Docs-Manage |
| 梳理实验文件夹 | Docs-Manage |
| 归档实验结果 | Docs-Manage，主控确认 |
| git 管理 | Docs-Manage |
| 未来论文框架 | Research-Planner + Docs-Manage |

## 9. Session 和 Tmux 管理

tmux 和 GPU 由主控 session 统一安排，Build-Run 执行。

建议约定：

```text
tmux 1：主线训练
tmux 2：评估或批量实验
tmux 3：备用 GPU 训练
```

每次启动训练前必须记录：

- tmux 窗口。
- GPU 编号。
- batch size。
- checkpoint 来源。
- 输出 run 目录。
- 是否训练后自动评估。
- 是否训练结束后关机。

训练中断时必须记录：

- 中断原因。
- 已完成 epoch。
- 当前 best checkpoint。
- 是否需要立即评估。
- 是否需要用新 batch 或新配置重跑。

## 10. 文件记忆分层

长期记忆不依赖对话窗口，依赖文件系统。

| 层级 | 文件 | 作用 |
|------|------|------|
| 工作需求 | `CC任务工作需求.md`、`CC工作模式.md` | 定义协作方式 |
| 项目规则 | `CLAUDE.md`、`AGENTS.md` | 固定项目和 agent 行为 |
| 实验主线 | `plan*.md` | 记录路线、阶段、结果和下一步 |
| 模型事实 | `model_registry.py` | 记录 checkpoint、指标、协议和继承关系 |
| 原始产物 | `/root/autodl-tmp/runs/` | 保存日志、JSON、图片和权重 |

当文件冲突时，处理顺序为：

1. 先查看 eval JSON 和训练日志。
2. 再查看 `model_registry.py`。
3. 再查看对应 plan。
4. 最后由主控 session 统一修正。

## 11. 决策规则

所有 subagent 输出必须回到主控 session 合并。最终判断遵守以下规则：

- 没有统一评估协议，不做最终结论。
- 没有 checkpoint lineage，不做公平对比。
- 没有评估方法版本，不写入最终注册表结论。
- 没有写入 plan 和 registry，不进入下一阶段。
- 消融实验没有写清唯一变化变量，不作为有效消融。
- 失败实验没有记录试过什么和为什么放弃，视为未完成闭环。
- 低成本验证优先于长训练。
- 路线失败要写清失败原因，防止重复实验。
- 周报和论文材料只使用已确认结果。
