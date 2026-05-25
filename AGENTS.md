# Repository Guidelines

## Project Structure & Module Organization
This repository is an umbrella workspace for PyTorch remote-sensing segmentation experiments. Root-level `README.md`, `plan*.md`, and `CLAUDE.md` describe research context and active findings. Our experiment code lives under `Personal-Project/` (RS-SAM3 through RS-SAM3-p8). Vendored reference projects under `Reference-Project/` (sam3-main, SegEarth-OV-3-main, MFNet, SAM3_LoRA-main, etc.). Plan documents under `Plan/`.

Use `utils/` for reusable image-processing helpers, `docs/` for paper notes and extracted materials, and `/root/autodl-tmp` for datasets, checkpoints, and run outputs. Do not commit large model weights, generated runs, or dataset tiles unless explicitly intended.

## Build, Test, and Development Commands
There is no single root build system. Run commands from the relevant subproject directory.

- `pip install -e Reference-Project/sam3-main/`: install the official SAM3 package for local imports.
- `python Reference-Project/MFNet/train.py`: run MFNet training, configured via `SSRS_*` environment variables.
- `python Reference-Project/SAM_RS/train.py`: train legacy SAM-assisted segmentation models.
- `cd Personal-Project/RS-SAM3-p4 && CUDA_VISIBLE_DEVICES=0 python train_full.py --dataset vaihingen --epochs 50 --batch 4`: run Plan4 full-training experiments.
- `cd sam3_isprs && python train_official.py`: run official SAM3-style ISPRS fine-tuning.
- `python utils/image_split.py` / `python utils/image_merge.py`: split and merge remote-sensing tiles.

## Coding Style & Naming Conventions
Write Python with 4-space indentation, `snake_case` functions and files, and `CamelCase` classes. Keep training scripts explicit and experiment-oriented; prefer clear argument names over hidden global behavior. Preserve existing folder conventions for variants, for example `train_*.py`, `eval_*.py`, `*_decoder.py`, and `configs/*.yaml`.

## Testing Guidelines
Tests are sparse and subproject-specific. Before changing model utilities or package code, run the nearest available checks, such as `python Reference-Project/sam3-main/test/test_io_utils.py`, `python Reference-Project/SAM3_LoRA-main/test_lora_injection.py`, or the relevant `test.py`. For training changes, prefer a short smoke run with reduced epochs or samples and record output paths under `/root/autodl-tmp/runs/`.

## Commit & Pull Request Guidelines
Git history uses short, descriptive messages rather than a strict conventional format, often including experiment or version labels such as `week5-end`, `0.2.1`, or `Remove model weights from tracking`. Keep commits focused and mention the affected experiment or module.

Pull requests should include the research goal, changed scripts/configs, exact commands run, dataset assumptions, key metrics, and output directory. Include screenshots or qualitative masks for visualization changes. Call out any checkpoint, dataset, or CUDA/PyTorch requirement needed to reproduce results.

## Agent-Specific Instructions
Respect existing research conclusions in `CLAUDE.md` and `plan*.md`. Avoid modifying vendored reference projects unless the task targets them directly. Keep generated artifacts, checkpoints, and datasets outside git-tracked source paths.

## Subagent Dispatch Templates

本项目采用 1 主控 + 4 subagent 模式。每个 subagent 的角色定义文件在 `.claude/agents/` 下。

### 通用任务包格式

分派 subagent 时使用固定格式：

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

### Subagent 类型与 prompt 前缀

**Research-Planner** — 研究规划
```
prompt: "你是 Research-Planner。请阅读 .claude/agents/research-planner.md 了解你的角色定义。

[任务包]

输出分析结果、plan 草案或消融链路表。"
```

**Build-Run** — 代码与训练
```
prompt: "你是 Build-Run。请阅读 .claude/agents/build-run.md 了解你的角色定义。

[任务包]

完成后报告修改文件列表、smoke test 结果和训练状态。"
```

**Eval-Analyze** — 评估分析
```
prompt: "你是 Eval-Analyze。请阅读 .claude/agents/eval-analyze.md 了解你的角色定义。

[任务包]

输出指标对比表、消融分析、推理图路径和可对齐性判断。"
```

**Docs-Manage** — 文档管理
```
prompt: "你是 Docs-Manage。请阅读 .claude/agents/docs-manage.md 了解你的角色定义。

[任务包]

输出修改文件列表、更新内容摘要和归档/git 检查结果。

归档任务特别注意事项：
- 禁止修改约束中，将"不删除 runs 文件"改为"不删除 runs 文件除非主控确认"
- 遇到磁盘空间不足时，立即报告主控而非自行拆分压缩包
- 建议方案：先删已归档的不重要权重 → 再继续打包"
```

### 使用方式

主控 session 通过 Agent 工具（subagent_type=general-purpose）分派任务，在 prompt 中包含角色文件引用和具体任务包。Subagent 的 isolation 默认使用 worktree 模式进行代码修改任务。

### Subagent 角色文件

| Subagent | 文件 | 核心职责 |
|----------|------|----------|
| Research-Planner | `.claude/agents/research-planner.md` | 论文解读、实验反思、plan 草案 |
| Build-Run | `.claude/agents/build-run.md` | 代码实现、检查、训练启动、GPU 管理 |
| Eval-Analyze | `.claude/agents/eval-analyze.md` | 正式评估、指标对齐、推理图生成 |
| Docs-Manage | `.claude/agents/docs-manage.md` | plan/注册表更新、周报、归档、git 管理 |
