"""
MFNet 协议模型注册表 — 严格对标 MFNet 论文 (IEEE TGRS 2025) 评估标准。
本注册表的数据可与 MFNet 论文 Table I (Vaihingen) 和 Table II (Potsdam) 直接对比。

═══════════════════════════════════════════════════════════
评估标准 (MFNet 论文协议)
═══════════════════════════════════════════════════════════

  参数          | Vaihingen             | Potsdam
  ─────────────┼──────────────────────┼─────────────────
  窗口大小       | 256×256              | 256×256
  步长 (stride)  | 32                   | 32
  边缘裁剪 (trim)| 0                    | 0
  标签           | 侵蚀 (eroded)        | 侵蚀 (noBoundary)
  Logit 累积     | soft-logit           | soft-logit
  DSM 归一化     | per-tile min-max     | per-tile min-max
  RGB 归一化     | /255.0               | /255.0
  批量推理       | ≥14                  | ≥14
  类别数 (评估)  | 5 (排除 clutter)     | 5 (排除 clutter)

  关键差异 vs 我们的内部 256² 协议 (model_registry.py):
  - stride: 32 vs 128 → 密集 ~16x 采样 → 结果高 ~9pp
  - trim: 0 vs 16 → 不裁剪边缘
  - labels: 侵蚀 vs 非侵蚀 → 边界像素被排除

  协议溯源:
  - stride=32: Reference-Project/MFNet/train.py line 532, 543 (硬编码)
  - no trim: Reference-Project/MFNet/utils.py sliding_window()
  - soft-logit: Reference-Project/MFNet/train.py test() 函数
  - 侵蚀标签: gts_eroded_for_participants (Vaihingen),
               5_Labels_for_participants_no_Boundary (Potsdam)
  - 批量推理: Reference-Project/MFNet/train.py test() 使用 grouper()

  详细文档: docs/MFNet_eval_protocol.md

═══════════════════════════════════════════════════════════
数据说明
═══════════════════════════════════════════════════════════

  per_class_oa / per_class_recall: 均为 Recall = TP/(TP+FN)
  与 MFNet 论文的 "per-class OA" 列同公式, 可直接对比。

  per_class_iou: TP/(TP+FP+FN) — 论文未直接报告, 但从混淆矩阵可推导。

  confusion_recall_view: 行归一化混淆矩阵 (% of GT → Pred),
  即 MFNet 论文 Table III 格式。

  数据集: ISPRS Vaihingen (12 train / 4 test), Potsdam (18 train / 6 test)
"""

# ═══════════════════════════════════════════════════════════
# MFNet 论文基线 (per_class_oa = Recall = TP/(TP+FN))
# ═══════════════════════════════════════════════════════════
MFNET_PAPER = {
    "Vaihingen": {
        "Frozen SAM1 ViT-L (no adapter)": {
            "oa": 88.01, "miou": 75.11, "mf1": 85.34,
            "per_class_recall": {"road": 89.51, "building": 94.64, "grass": 71.71, "tree": 89.47, "car": 76.83},
        },
        "MMLoRA ViT-L": {
            "oa": 91.31, "miou": 83.20, "mf1": 90.56,
            "per_class_recall": {"road": 93.27, "building": 97.21, "grass": 78.24, "tree": 91.02, "car": 87.18},
        },
        "MMAdapter ViT-L": {
            "oa": 92.02, "miou": 83.69, "mf1": 90.94,
            "per_class_recall": {"road": 92.59, "building": 96.29, "grass": 80.15, "tree": 93.09, "car": 89.08},
        },
        "MMAdapter ViT-H (best)": {
            "oa": 92.93, "miou": 84.72, "mf1": 91.51,
            "per_class_recall": {"road": 93.39, "building": 98.84, "grass": 81.16, "tree": 93.17, "car": 89.23},
        },
    },
    "Potsdam": {
        "MMLoRA ViT-L": {
            "oa": 90.99, "miou": 85.71, "mf1": 92.13,
            "per_class_recall": {"road": 92.68, "building": 97.59, "grass": 88.34, "tree": 88.57, "car": 96.35},
        },
        "MMAdapter ViT-L": {
            "oa": 91.62, "miou": 86.37, "mf1": 92.51,
            "per_class_recall": {"road": 93.69, "building": 98.31, "grass": 87.27, "tree": 88.78, "car": 96.29},
        },
        "MMAdapter ViT-H (best)": {
            "oa": 91.71, "miou": 86.69, "mf1": 92.70,
            "per_class_recall": {"road": 93.17, "building": 98.44, "grass": 90.36, "tree": 87.37, "car": 96.24},
        },
    }
}

# ═══════════════════════════════════════════════════════════
# 我们的模型 (MFNet 协议评估)
# ═══════════════════════════════════════════════════════════
OUR_RESULTS = [
    # ── Vaihingen ──
    {
        "name": "Plan7-A",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_225309/best_model.pt",
        "architecture": "SAM3 ViTDet + in-ViT MMAdapter (32 blocks, bn=32) + 3-way gate + DSM edge/slope prompt + MFNetDecoder",
        "init_from": "Plan6 Phase1 (热启动, epoch 1)",
        "trainable_params": "~12.7M / 820M (1.55%)",
        "eval_timestamp": "2026-06-05",

        "oa": 92.87, "miou": 86.22, "mean_recall": 93.30,
        "per_class_iou":    {"road": 87.76, "building": 95.19, "grass": 72.00, "tree": 86.00, "car": 90.12},
        "per_class_recall": {"road": 92.47, "building": 97.45, "grass": 84.33, "tree": 93.09, "car": 98.70},
        "per_class_precision": {"road": 94.37, "building": 97.57, "grass": 83.10, "tree": 91.76, "car": 91.26},
        "per_class_f1":    {"road": 93.41, "building": 97.51, "grass": 83.71, "tree": 92.42, "car": 94.83},

        "confusion_recall_view": {
            "road":     {"road": 92.5, "building": 1.9, "grass": 4.1, "tree": 0.7, "car": 0.9},
            "building": {"road": 1.8, "building": 97.5, "grass": 0.6, "tree": 0.1, "car": 0.0},
            "grass":    {"road": 4.4, "building": 0.6, "grass": 84.3, "tree": 10.2, "car": 0.5},
            "tree":     {"road": 0.7, "building": 0.1, "grass": 6.1, "tree": 93.1, "car": 0.1},
            "car":      {"road": 1.2, "building": 0.0, "grass": 0.0, "tree": 0.1, "car": 98.7},
        },
        "key_confusion": {"grass→tree": 10.2, "tree→grass": 6.1, "veg_sum": 16.3},

        "vs_mfnet_best": {
            "delta_miou": +1.50, "delta_oa": -0.06,
            "delta_recall": {"road": -0.92, "building": -1.39, "grass": +3.17, "tree": -0.08, "car": +9.47},
        },
        "note": "car recall 大幅超越 MFNet (+9.47pp)。grass recall 也超出 +3.17pp。road/building 略低于 MFNet，tree 持平。"
    },

    # ── Potsdam ──
    {
        "name": "P10-A (Plan7-A on Potsdam)",
        "dataset": "Potsdam",
        "ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_potsdam_20260526_210230/best_model.pt",
        "architecture": "SAM3 ViTDet + in-ViT MMAdapter (32 blocks, bn=32) + 3-way gate + DSM edge/slope prompt + MFNetDecoder",
        "init_from": "from scratch (20 epoch, seed=default)",
        "trainable_params": "~12.7M / 820M (1.55%)",
        "eval_timestamp": "2026-06-05",

        "oa": 94.26, "miou": 89.20, "mean_recall": 94.38,
        "per_class_iou":    {"road": 91.96, "building": 97.23, "grass": 81.59, "tree": 80.18, "car": 95.04},
        "per_class_recall": {"road": 94.79, "building": 98.57, "grass": 89.84, "tree": 90.96, "car": 97.72},
        "per_class_precision": {"road": 96.90, "building": 98.63, "grass": 89.68, "tree": 87.11, "car": 97.18},
        "per_class_f1":    {"road": 95.84, "building": 98.60, "grass": 89.76, "tree": 88.99, "car": 97.45},

        "confusion_recall_view": {
            "road":     {"road": 94.9, "building": 0.7, "grass": 3.1, "tree": 1.3, "car": 0.0},
            "building": {"road": 0.8, "building": 98.6, "grass": 0.3, "tree": 0.3, "car": 0.0},
            "grass":    {"road": 2.3, "building": 0.6, "grass": 89.8, "tree": 7.3, "car": 0.0},
            "tree":     {"road": 1.5, "building": 0.3, "grass": 7.1, "tree": 90.9, "car": 0.2},
            "car":      {"road": 0.3, "building": 0.2, "grass": 0.3, "tree": 1.5, "car": 97.7},
        },
        "key_confusion": {"grass→tree": 7.3, "tree→grass": 7.1, "veg_sum": 14.4},

        "vs_mfnet_best": {
            "delta_miou": +2.51, "delta_oa": +2.55,
            "delta_recall": {"road": +1.62, "building": +0.13, "grass": -0.52, "tree": +3.59, "car": +1.48},
        },
        "note": "adapter+prompt 架构在 Potsdam 上表现强劲。tree recall 超出 MFNet Best +3.59pp。"
    },
    {
        "name": "P10-B (Phase4 F0 on Potsdam)",
        "dataset": "Potsdam",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_potsdam_20260527_092853/best_model.pt",
        "architecture": "SAM3 ViTDet + LoRA rank=8 (shared encoder) + 1×SEFusion + MFNetDecoder",
        "init_from": "from scratch (20 epoch, seed=42)",
        "trainable_params": "~5.8M / 814M (0.72%)",
        "eval_timestamp": "2026-06-05",

        "oa": 94.22, "miou": 89.18, "mean_recall": 94.30,
        "per_class_iou":    {"road": 91.83, "building": 96.75, "grass": 81.87, "tree": 80.29, "car": 95.18},
        "per_class_recall": {"road": 94.36, "building": 98.62, "grass": 92.10, "tree": 88.33, "car": 98.09},
        "per_class_precision": {"road": 97.15, "building": 98.08, "grass": 87.93, "tree": 89.31, "car": 96.99},
        "per_class_f1":    {"road": 95.74, "building": 98.35, "grass": 89.96, "tree": 88.82, "car": 97.54},

        "confusion_recall_view": {
            "road":     {"road": 94.4, "building": 1.1, "grass": 3.4, "tree": 1.1, "car": 0.0},
            "building": {"road": 0.7, "building": 98.6, "grass": 0.6, "tree": 0.2, "car": 0.0},
            "grass":    {"road": 1.9, "building": 0.6, "grass": 92.1, "tree": 5.4, "car": 0.0},
            "tree":     {"road": 1.7, "building": 0.4, "grass": 9.4, "tree": 88.3, "car": 0.2},
            "car":      {"road": 0.4, "building": 0.4, "grass": 0.2, "tree": 1.0, "car": 98.1},
        },
        "key_confusion": {"grass→tree": 5.4, "tree→grass": 9.4, "veg_sum": 14.8},

        "vs_mfnet_best": {
            "delta_miou": +2.49, "delta_oa": +2.51,
            "delta_recall": {"road": +1.19, "building": +0.18, "grass": +1.74, "tree": +0.96, "car": +1.85},
        },
        "note": "grass recall 最高 (92.10%, +4.83pp vs MFNet-ViT-L) 但 tree recall 最低 (88.33%)。LoRA rank=8 + 1×SEFusion 偏向 grass 纹理，tree→grass 混淆达 9.4%。"
    },
    {
        "name": "P10-C (F0'+L on Potsdam)",
        "dataset": "Potsdam",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_potsdam_20260528_105807/best_model.pt",
        "architecture": "SAM3 ViTDet + LoRA rank=8 (attn+MLP, frozen encoder) + 4×SEFusion + MFNetDecoder",
        "init_from": "F0' frozen ckpt → 20 epoch continuation",
        "trainable_params": "~7.2M",
        "eval_timestamp": "2026-06-05",

        "oa": 94.30, "miou": 89.35, "mean_recall": 94.50,
        "per_class_iou":    {"road": 92.00, "building": 96.58, "grass": 82.08, "tree": 80.70, "car": 95.39},
        "per_class_recall": {"road": 94.44, "building": 98.72, "grass": 90.23, "tree": 91.02, "car": 98.10},
        "per_class_precision": {"road": 97.27, "building": 97.82, "grass": 89.78, "tree": 87.59, "car": 97.18},
        "per_class_f1":    {"road": 95.84, "building": 98.27, "grass": 90.00, "tree": 89.27, "car": 97.64},

        "confusion_recall_view": {
            "road":     {"road": 94.4, "building": 1.4, "grass": 2.8, "tree": 1.4, "car": 0.0},
            "building": {"road": 0.6, "building": 98.7, "grass": 0.4, "tree": 0.3, "car": 0.0},
            "grass":    {"road": 2.1, "building": 0.8, "grass": 90.2, "tree": 6.9, "car": 0.0},
            "tree":     {"road": 1.2, "building": 0.3, "grass": 7.3, "tree": 91.0, "car": 0.2},
            "car":      {"road": 0.3, "building": 0.3, "grass": 0.1, "tree": 1.2, "car": 98.1},
        },
        "key_confusion": {"grass→tree": 6.9, "tree→grass": 7.3, "veg_sum": 14.2},

        "vs_mfnet_best": {
            "delta_miou": +2.66, "delta_oa": +2.59,
            "delta_recall": {"road": +1.27, "building": +0.28, "grass": -0.13, "tree": +3.65, "car": +1.86},
        },
        "note": "Potsdam 最佳。4×SEFusion 比 1×SEFusion 更平衡: tree→grass 从 9.4% 降至 7.3%，仅牺牲 1.9pp grass recall。vegetation confusion 最平衡 (14.2%)。"
    },

    # ── Plan11 系列 (Vaihingen, MFNet 协议) ──
    {
        "name": "P11-A (min-max DSM + veg boundary weight, from Plan7-A)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan11_a_veg_boundary_loss_vaihingen_20260604_202157/best_model.pt",
        "architecture": "Plan7PromptMFNet (Plan7-A arch, min-max DSM) + veg_boundary_weight=3.0",
        "init_from": "Plan7-A (continuation)",
        "eval_timestamp": "2026-06-07",

        "oa": 89.32, "miou": 81.19,
        "per_class_iou":    {"roads": 79.6, "buildings": 86.5, "low veg.": 66.5, "trees": 84.7, "cars": 88.7},
        "note": "min-max DSM + veg boundary weight。grass→tree 15.1%。与 P11-B 不可直接对比 (DSM 归一化不同, 非 nDSM)。"
    },
    {
        "name": "P11-C (nDSM + veg boundary weight, from scratch)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan11_c_combined_vaihingen_20260605_012150/best_model.pt",
        "architecture": "Plan7PromptMFNet + nDSM + veg_boundary_weight=3.0",
        "init_from": "from scratch (seed=42, 15 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.15, "miou": 85.15,
        "per_class_iou":    {"roads": 86.6, "buildings": 93.6, "low veg.": 70.5, "trees": 85.4, "cars": 89.8},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": -0.16},
        "note": "-0.16pp vs P11-B (噪声级)。veg boundary weight 在 nDSM 下仍然无效。"
    },
    {
        "name": "P11-D (nDSM + photometric augmentation, from scratch)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan11_d_aug_vaihingen_20260605_035155/best_model.pt",
        "architecture": "Plan7PromptMFNet + nDSM + ColorJitter(0.2) + GaussianBlur(0.3)",
        "init_from": "from scratch (seed=42, 11 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.33, "miou": 85.35,
        "per_class_iou":    {"roads": 87.0, "buildings": 93.8, "low veg.": 70.7, "trees": 85.5, "cars": 89.7},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": +0.04},
        "note": "+0.04pp vs P11-B (噪声级)。256² 下 -0.16pp, MFNet 下 +0.04pp — 方向不一致, 确认为噪声。"
    },
    {
        "name": "P11-E (nDSM + 4-block adapter only, from scratch)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan11_e_adapter_vaihingen_20260605_062329/best_model.pt",
        "architecture": "Plan7PromptMFNet + nDSM + adapter bneck=8, 4 global blocks [7,15,23,31] only",
        "init_from": "from scratch (seed=42, 15 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 87.93, "miou": 79.21,
        "per_class_iou":    {"roads": 83.0, "buildings": 91.3, "low veg.": 60.5, "trees": 72.9, "cars": 88.3},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": -6.10},
        "note": "-6.10pp vs P11-B。adapter 缩减到 4 blocks 且 bneck=8 导致严重退化。"
    },

    # ── Plan13 系列 (Vaihingen, MFNet 协议) ──
    {
        "name": "P11-B (nDSM baseline, Plan13 ref)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_20260604_225146/best_model.pt",
        "architecture": "SAM3 ViTDet + in-ViT MMAdapter (32 blocks, bn=32) + 3-way gate + DSM prompt + MFNetDecoder, nDSM",
        "init_from": "from scratch (seed=42, 15 epoch)",
        "trainable_params": "~12.7M / 820M (1.55%)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.28, "miou": 85.31,
        "per_class_iou":    {"roads": 86.3, "buildings": 93.2, "low veg.": 71.5, "trees": 86.1, "cars": 89.5},
        "per_class_recall": {"roads": 94.3, "buildings": 95.3, "low veg.": 82.6, "trees": 92.5, "cars": 98.1},
        "note": "Plan13 系列 MFNet 协议基线。256² stride=128 下 76.72%。"
    },
    {
        "name": "P13-A (Multi-level ViT feature pyramid)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan13_a_multilevel_vit_vaihingen_20260605_164138/best_model.pt",
        "architecture": "P11-B + 4×1×1 conv (block7→1/4, block15→1/8, block23→1/16, block31→1/32) → MFNetDecoder",
        "init_from": "from scratch (seed=42, 15 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.05, "miou": 84.84,
        "per_class_iou":    {"roads": 86.1, "buildings": 92.7, "low veg.": 71.1, "trees": 85.6, "cars": 88.7},
        "per_class_recall": {"roads": 94.0, "buildings": 94.9, "low veg.": 82.6, "trees": 92.4, "cars": 98.2},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": -0.47},
        "note": "-0.47pp vs P11-B。ViT 14×14 patch 瓶颈。方向关闭。"
    },
    {
        "name": "P13-C (RGB texture branch + vegetation refinement)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan13_c_texture_branch_vaihingen_20260605_215434/best_model.pt",
        "architecture": "P11-B + TextureStem + VegRefinementHead (2ch tree/grass delta, full-image gating)",
        "init_from": "P11-B (continuation, 10 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.35, "miou": 85.55,
        "per_class_iou":    {"roads": 86.4, "buildings": 92.9, "low veg.": 71.9, "trees": 86.4, "cars": 90.2},
        "per_class_recall": {"roads": 94.0, "buildings": 95.5, "low veg.": 82.4, "trees": 93.1, "cars": 97.9},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": +0.24},
        "note": "+0.24pp vs P11-B。tree recall +0.6pp，有方向性信号但净效应噪声级。"
    },
    {
        "name": "P13-E (Gated veg delta + building suppression)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan13_e_building_suppression_vaihingen_20260605_232903/best_model.pt",
        "architecture": "P13-C + single-channel veg confidence gate (sigmoid) + building suppression loss",
        "init_from": "P13-C (continuation, 10 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.28, "miou": 85.53,
        "per_class_iou":    {"roads": 86.0, "buildings": 92.9, "low veg.": 71.9, "trees": 86.4, "cars": 90.5},
        "per_class_recall": {"roads": 94.5, "buildings": 95.1, "low veg.": 82.8, "trees": 92.4, "cars": 97.9},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": +0.22},
        "note": "+0.22pp vs P11-B。共享 gate 过度抑制所有 veg。grass recall 最高 (82.8%)，tree recall 最低 (92.4%)。"
    },
    {
        "name": "P13-F (Per-class vegetation gates + cross-suppression)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan13_f_perclass_gate_vaihingen_20260606_130419/best_model.pt",
        "architecture": "P13-C + 2 separate gates (tree_gate + grass_gate) + cross-suppression loss",
        "init_from": "P13-C (continuation, 10 epoch)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.33, "miou": 85.61,
        "per_class_iou":    {"roads": 86.3, "buildings": 92.9, "low veg.": 71.9, "trees": 86.3, "cars": 90.7},
        "per_class_recall": {"roads": 93.4, "buildings": 95.5, "low veg.": 82.2, "trees": 93.8, "cars": 97.6},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": +0.30},
        "note": "Plan13 全图 gate 最佳 (+0.30pp)。tree recall 93.8% 最高但 road -0.95pp。全图 gate = 零和重排序。方向关闭。"
    },
    {
        "name": "P13-G5 (Strict veg-only multiscale texture + nDSM roughness + DiceCE)",
        "dataset": "Vaihingen",
        "ckpt": "/root/autodl-tmp/runs/plan13_g_g5_pred_rgb_ndsmrough_dicece_vaihingen_20260606_232552/best_model.pt",
        "architecture": "P11-B frozen + MultiScaleTextureStem (3×3/7×7/11×11) + NDSMRoughnessStem + VegetationBinaryHead, strict veg mask",
        "init_from": "P11-B (continuation, 10 epoch, only head trained)",
        "eval_timestamp": "2026-06-07",

        "oa": 92.30, "miou": 85.30,
        "per_class_iou":    {"roads": 86.2, "buildings": 93.2, "low veg.": 71.4, "trees": 86.1, "cars": 89.5},
        "per_class_recall": {"roads": 94.5, "buildings": 95.3, "low veg.": 81.3, "trees": 93.3, "cars": 98.1},
        "vs_baseline": {"baseline": "P11-B", "delta_miou": -0.01},
        "note": "-0.01pp vs P11-B。与 256² +0.48pp 排名反转 — eroded labels 排除了 P13-G5 改善集中的边界像素。256² 协议下是 Plan13 唯一正向结果。"
    },
]

# ═══════════════════════════════════════════════════════════
# 与 MFNet 论文最终对比摘要
# ═══════════════════════════════════════════════════════════
COMPARISON_SUMMARY = {
    "Vaihingen": {
        "our_best": "Plan7-A",
        "our_miou": 86.22, "mfnet_best_miou": 84.72, "delta_miou": +1.50,
        "our_oa": 92.87, "mfnet_best_oa": 92.93, "delta_oa": -0.06,
        "per_class_recall_delta": {"road": -0.92, "building": -1.39, "grass": +3.17, "tree": -0.08, "car": +9.47},
        "our_strengths": ["car (+9.47pp)", "grass (+3.17pp)"],
        "our_weaknesses": ["building (-1.39pp)", "road (-0.92pp)"],
    },
    "Potsdam": {
        "our_best": "P10-C (F0'+L)",
        "our_miou": 89.35, "mfnet_best_miou": 86.69, "delta_miou": +2.66,
        "our_oa": 94.30, "mfnet_best_oa": 91.71, "delta_oa": +2.59,
        "per_class_recall_delta": {"road": +1.27, "building": +0.28, "grass": -0.13, "tree": +3.65, "car": +1.86},
        "our_strengths": ["tree (+3.65pp)", "car (+1.86pp)", "road (+1.27pp)"],
        "our_weaknesses": ["grass (-0.13pp, 微弱)"],
    },
}
