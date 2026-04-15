# Week5 联合框架验证执行计划（基于当前 Week4 进展）

## 1. 当前输入（截至 2026-04-13）
- Week4 Day1 最优：c05_l0.0030_g1.0_a0.0010，best mIoU 0.5441（seed42）。
- Week4 Day2 最优：c06_l0.0020_g1.5_a0.0015，best mIoU 0.5273（seed42）。
- 当前缺口：尚未完成 Week4 多种子统计与 single-level vs multi-level 对齐表。

## 2. Week5 总目标（与排期表对齐）
- 在稳定 UDA 主线基础上验证结构约束（BDY/OBJ）是否保留。
- 形成 T2-v3、F3-v2、F5-v1 三项可交付结果。
- 将 prompt 分支保持为条件增强，不抢占主线资源。

## 3. 执行优先级（按顺序）
1. 先补齐 Week4 缺口，锁定 Week5 主配置。
2. 再做 Week5 主表实验（Source-only / +UDA / +UDA+BDY / +UDA+OBJ / +UDA+BDY+OBJ）。
3. 最后做边界放大图与伪标签质量图。

## 4. 本周实验矩阵
- 固定协议：Vaihingen -> Potsdam。
- 推荐主配置候选：
  - A: c06_l0.0020_g1.5_a0.0015
  - B: c05_l0.0030_g1.0_a0.0010
- 种子策略：
  - 快速决策：seed 42（先跑全矩阵）
  - 稳定性确认：seed 7/42/3407（至少补到 2 seed，最好 3 seed）

## 5. 建议时间表（1 GPU 场景）
- Day5-1:
  - 补 Week4：A/B 两组的多 seed（至少 2 seed）
  - 输出 mean±std 临时表，确定 Week5 主配置
- Day5-2 到 Day5-3:
  - 运行 Week5 主表五组：
    - Source-only
    - +UDA
    - +UDA+BDY
    - +UDA+OBJ
    - +UDA+BDY+OBJ
- Day5-4:
  - 生成 F3-v2（边界放大）与 F5-v1（伪标签质量/边界响应）
- Day5-5:
  - 汇总 T2-v3，并补失败案例说明

## 6. 目录与命名规范（建议）
- runs/week5_joint/
  - 20260413_xxxxxx/
    - source_only_seed42/
    - uda_seed42/
    - uda_bdy_seed42/
    - uda_obj_seed42/
    - uda_bdy_obj_seed42/
    - summary_t2_v3.md
    - visual_f3_v2/
    - pseudo_f5_v1/

## 7. 立即可执行命令（现有脚本）
- Week4 多 seed（先补齐稳定性证据）：
  - bash utils/run_week4_day3_multiseed.sh
- Week4 收敛图（已完成，可复跑）：
  - python utils/generate_week4_f4_v3.py

## 8. 风险与止损规则
- 若 +UDA+BDY 或 +UDA+OBJ 在 2 个 seed 下均不优于 +UDA：
  - 该结构项降级为分析分支，不进入主模型。
- 若 +UDA+BDY+OBJ 波动显著大于 +UDA：
  - 优先保留单结构项（BDY 或 OBJ）而非双结构叠加。
- 若 prompt 分支当周无法形成稳定正增益：
  - 保持附录定位，不进入正文主贡献。

## 9. 本周最低交付线
- T2-v3：5 组主结果表（至少单 seed 完整，最好含 2-seed 统计）。
- F3-v2：2 到 3 个典型 patch 的边界放大图。
- F5-v1：至少 1 组伪标签质量或边界响应可视化。
