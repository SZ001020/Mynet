"""
模型注册表：所有实验阶段的模型、checkpoint 和评估结果。

Plan1: Zero-shot baseline + Prompt engineering（无训练 checkpoint，仅参考基线）
Plan2: Per-class binary paradigm + Threshold tuning（无训练 checkpoint）
Plan3: Route A VPT / Route B LoRA（有 checkpoint）
Plan4: Full training SGD / Unfreeze（有 checkpoint）
Plan5: Boundary/object auxiliary supervision（有 checkpoint）
Plan6: In-ViT RGB/DSM MMAdapter（有 checkpoint）
Plan7: 多模态结构先验注入（Phase A 已完成，有 checkpoint）

═══════════════════════════════════════════════════════════
评估标准 (Evaluation Protocol)
═══════════════════════════════════════════════════════════

本注册表使用「256² 内部协议」，与 MFNet 论文协议不同：

  参数          | 本注册表 (256² 协议)   | MFNet 论文协议
  ─────────────┼──────────────────────┼─────────────────
  窗口大小       | 256×256              | 256×256
  步长 (stride)  | 128                  | 32
  边缘裁剪 (trim)| 16                   | 0
  标签           | 非侵蚀 (participants) | 侵蚀 (noBoundary)
  Logit 累积     | per-patch argmax 或 soft-logit | soft-logit
  DSM 归一化     | per-tile min-max 或 nDSM | per-tile min-max
  批量推理       | 1                    | 14

典型差异: 同一模型用 MFNet 协议评估会比本协议高 ~9pp mIoU。
本注册表的数据不可直接与 MFNet 论文数字对比。
如需与 MFNet 论文对比, 请使用 mfnet_protocol_registry.py。

各条目通过 protocol 字段或 note 字段标注所用协议:
  - "per-patch argmax": 早期 per-patch argmax 评估 (低估 ~0.4-0.8pp)
  - "soft-logit accumulation": 标准软 logit 累积 (推荐)
  - "global confusion matrix": 全局混淆矩阵累积

═══════════════════════════════════════════════════════════
数据说明
═══════════════════════════════════════════════════════════

⚠️ eval_per_class_oa 说明：
  - 2026-05-11 之前，所有 per_class_oa 使用 (TP+TN)/total 公式，car/grass 会被 TN 大幅拉高。
  - 2026-05-11 起，新增 per_class_recall = TP/(TP+FN)，与 MFNet 论文的 "per-class OA" 公式一致。
  - 有 per_class_recall 的条目才可跨模型对比；仅有 per_class_oa 的条目使用旧公式，不可对比。

⚠️ 2026-05-12 评估聚合口径更新：
  - 正式 eval 脚本改为全验证集累计 correct/intersection/union 后计算 OA、mIoU、per-class recall。
  - 旧 JSON/注册表中已经保存的 OA/mIoU 多数来自 tile-wise 简单平均，数值可能有小幅差异。

用法:
  python eval_universal.py --list    # 列出所有模型
  python eval_universal.py           # 评估所有可加载模型
"""

REGISTRY = [
    # ═══════════════════════════════════════════════════════════
    # Plan1 — 零样本基线（256² 统一标准重评估 2026-05-09）
    #   ⚠️ 原 Plan1 整图单次推理高估了 ~5.5pp mIoU
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan1 Zero-shot (整图, 原协议) — Vaihingen",
        "phase": "Plan1 Phase 1 (original)",
        "ckpt": None,
        "source": "Reference-Project/SegEarth-OV-3-main",
        "protocol": "full-tile single-view, max 2000px",
        "eval_oa": None, "eval_miou": 66.0, "eval_mf1": None,
        "note": "原 Plan1 协议。整图单次推理 → argmax。被 256² 协议取代。"
    },
    {
        "name": "Plan1 Zero-shot (256² 统一标准) — Vaihingen",
        "phase": "Plan1 Phase 1 (re-eval)",
        "ckpt": None,
        "source": "Personal-Project/RS-SAM3-p1",
        "protocol": "256² sliding window, stride=128",
        "eval_oa": 77.84, "eval_miou": 60.46, "eval_mf1": None,
        "eval_per_class_iou": {"road": 64.3, "building": 71.6, "grass": 37.9, "tree": 67.8, "car": 60.8},
        "eval_per_class_oa": {"road": 88.5, "building": 90.4, "grass": 86.3, "tree": 91.0, "car": 99.4},
        "note": "统一标准重评估。car IoU +9pp vs 整图(51.8→60.8)，building -14.7pp(86.3→71.6)。"
    },
    {
        "name": "Plan1 Zero-shot Dual-Head (Potsdam, 原协议)",
        "phase": "Plan1 Phase 1 (original)",
        "ckpt": None,
        "source": "Reference-Project/SegEarth-OV-3-main",
        "oa": None, "miou": 56.6, "mf1": None,
        "note": "SAM3 zero-shot on Potsdam, 整图单次推理。待重评估。"
    },
    {
        "name": "Plan1 Phase 3 FPN fine-tune",
        "phase": "Plan1 Phase 3",
        "ckpt": None,  # deleted in cleanup
        "source": "Reference-Project/fineNet",
        "oa": None, "miou": 56.6, "mf1": None,
        "note": "Frozen SAM3 + FPN decoder (797K). Checkpoint已删除，无法重评估。"
    },

    # ═══════════════════════════════════════════════════════════
    # Plan2 — Per-class binary paradigm（256² 统一标准重评估待完成）
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan2 Per-class Binary (原协议, untuned) — Vaihingen",
        "phase": "Plan2 Phase 1 (original)",
        "ckpt": None,
        "source": "Personal-Project/RS-SAM3",
        "protocol": "full-tile single-view, per-class binary top-15%",
        "eval_oa": None, "eval_miou": 45.0, "eval_mf1": None,
        "note": "逐类二值评估（非多类别mIoU），与多类别指标不可直接对比。"
    },
    {
        "name": "Plan2 Threshold Tuned (原协议) — Vaihingen",
        "phase": "Plan2 Phase 1b (original)",
        "ckpt": None,
        "source": "Personal-Project/RS-SAM3",
        "protocol": "full-tile single-view, per-class tuned K%",
        "eval_oa": None, "eval_miou": 55.4, "eval_mf1": None,
        "note": "逐类二值评估 + per-class 阈值调优。"
    },
    {
        "name": "Plan2 (256² 统一标准) — Vaihingen",
        "phase": "Plan2 (re-eval 2026-05-09)",
        "ckpt": None,
        "source": "Personal-Project/RS-SAM3-p2",
        "protocol": "256² sliding window, stride=128, instance OR semantic top-K%",
        "eval_oa": 68.19, "eval_miou": 49.76, "eval_mf1": None,
        "eval_per_class_iou": {"road": 53.7, "building": 75.7, "grass": 23.7, "tree": 48.3, "car": 47.4},
        "eval_per_class_oa": {"road": 80.5, "building": 92.3, "grass": 78.2, "tree": 86.4, "car": 99.0},
        "eval_binary_iou": {"road": 45.7, "building": 72.3, "grass": 25.5, "tree": 46.0, "car": 28.0},
        "best_k": {"road": 40, "building": 10, "grass": 30, "tree": 20, "car": 3},
        "note": "Plan1 dual-head max 比此方法高 10.7pp mIoU。硬二值化 OR 丢失概率信息。building 例外(+4.1pp)。"
    },

    # ═══════════════════════════════════════════════════════════
    # Plan3 Route A — VPT Adapter + UNet Decoder (Personal-Project/RS-SAM3-p3/p3r)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Route A RGB (VPT + simple UNet)",
        "phase": "Plan3 Route A",
        "ckpt": "/root/autodl-tmp/archives/plan3.tar.gz",
        "archived_note": "权重已归档至 plan3.tar.gz，原始 .pt 已删除。512² crop training. First model to beat zero-shot (65.8%).",
        "source": "Personal-Project/RS-SAM3-p3",
        "class": "AdapterSAM3UNet",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5},
        "use_dsm": False,
        "eval_oa": None, "eval_miou": 68.70, "eval_mf1": None
    },
    {
        "name": "Route A+DSM concat (VPT + simple UNet)",
        "phase": "Plan3 Route A",
        "ckpt": None,
        "deleted_note": "权重文件已删除（训练脚本存在但产出已被超越），可用 Personal-Project/RS-SAM3-p3/train_dual.py 复现",
        "source": "Personal-Project/RS-SAM3-p3",
        "class": "AdapterSAM3UNetDSM",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": 70.04, "eval_mf1": None,
        "note": "Simple concat DSM fusion. +1.3pp over RGB-only."
    },

    # Plan3 Route A — p3r (improved decoder + cross-attn DSM)
    {
        "name": "Route A+DSM cross-attn (VPT + UNetFormer)",
        "phase": "Plan3 Route A (p3r)",
        "ckpt": None,
        "deleted_note": "权重已删除，可用 RS-SAM3-p3r/train_256.py 复现",
        "source": "Personal-Project/RS-SAM3-p3r",
        "class": "AdapterSAM3UNetFormerDSM",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": 70.39, "eval_mf1": None,
        "note": "UNetFormer GLA decoder + cross-attn DSM. Best Route A (512² crop)."
    },
    {
        "name": "VPT+DSM UNetFormer (256² window best)",
        "phase": "Plan3 Route A (p3r)",
        "ckpt": None,
        "deleted_note": "权重已删除（被 MFNetDecoder 超越），可用 RS-SAM3-p3r/train_256.py 复现",
        "source": "Personal-Project/RS-SAM3-p3r",
        "class": "AdapterSAM3UNetFormerDSM",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.54, "eval_miou": 72.97, "eval_mf1": None,
        "note": "First 256² window training. +2.6pp over 512² crop training."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan3 Route B — LoRA on SAM3 ViT (RS-SAM-p3b)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "LoRA RGB (512² crop)",
        "phase": "Plan3 Route B",
        "ckpt": None,  # superseded
        "source": "Personal-Project/RS-SAM-p3b",
        "eval_oa": None, "eval_miou": 69.70, "eval_mf1": None,
        "note": "LoRA r=8 on ViT + UNetFormer. VPT > LoRA on frozen SAM3."
    },
    {
        "name": "LoRA RGB (256² window training)",
        "phase": "Plan3 Route B",
        "ckpt": "/root/autodl-tmp/archives/plan3.tar.gz",
        "archived_note": "权重已归档至 plan3.tar.gz，原始 .pt 已删除。256² window training. +3.2pp over 512² crop. Nearly tied with VPT+DSM.",
        "source": "Personal-Project/RS-SAM-p3b",
        "class": "LoRASAM3UNetFormer",
        "module": "lora_sam3",
        "kwargs": {"lora_rank": 8, "lora_alpha": 16, "num_classes": 5, "dropout": 0.1},
        "use_dsm": False,
        "eval_oa": 86.15, "eval_miou": 72.87, "eval_mf1": None
    },
    {
        "name": "LoRA+DSM (256² window training)",
        "phase": "Plan3 Route B",
        "ckpt": None,
        "deleted_note": "权重已删除（被 LoRA RGB 256² 超越），可用 RS-SAM3-p3r/train_256.py 复现",
        "source": "Personal-Project/RS-SAM-p3b",
        "class": "LoRASAM3UNetFormer",
        "module": "lora_sam3",
        "kwargs": {"lora_rank": 8, "lora_alpha": 16, "num_classes": 5, "dropout": 0.1, "use_dsm": True},
        "use_dsm": True,
        "eval_oa": 86.34, "eval_miou": 72.65, "eval_mf1": None,
        "note": "DSM adds minimal gain for LoRA. 17.3M params."
    },

    # Plan3 MFNet Decoder (RS-SAM-p3b)
    {
        "name": "VPT + MFNet Decoder (GLOBAL BEST, 73.10%)",
        "phase": "Plan3 Route A (final)",
        "ckpt": "/root/autodl-tmp/archives/plan3.tar.gz",
        "archived_note": "权重已归档至 plan3.tar.gz，原始 .pt 已删除。GLOBAL BEST. VPT 2.1M + MFNet Decoder 0.5M + DSM. 4.5M trainable params.",
        "source": "Personal-Project/RS-SAM-p3b",
        "class": "VPT_MFNetDecoder",
        "module": "train_mfnet_decoder",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.45, "eval_miou": 73.10, "eval_mf1": 84.50,
        "eval_per_class_iou": {"road": 74.6, "building": 85.8, "grass": 60.2, "tree": 76.0, "car": 68.9},
        "eval_per_class_oa": {"road": 89.2, "building": 90.5, "grass": 78.3, "tree": 84.7, "car": 76.4}
    },

    # ═══════════════════════════════════════════════════════════
    # Plan4 — Full training / Unfreeze (RS-SAM3-p4)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan4 D1 (unfreeze 8 layers, AdamW)",
        "phase": "Plan4 D1",
        "ckpt": None,
        "deleted_note": "权重已删除（失败路线），可用 RS-SAM3-p4/train_unfreeze.py 复现",
        "source": "Personal-Project/RS-SAM3-p4",
        "class": "VPT_MFNetDecoder_Unfreeze",
        "module": "train_unfreeze",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True,
                   "dropout": 0.1, "unfreeze_layers": 8},
        "use_dsm": True,
        "eval_oa": 86.39, "eval_miou": 72.48, "eval_mf1": None,
        "note": "116M params. Overfit at epoch 2. Worse than frozen (73.10%)."
    },
    {
        "name": "Plan4 D2 (unfreeze 16 layers, AdamW)",
        "phase": "Plan4 D2",
        "ckpt": None,
        "deleted_note": "权重已删除（失败路线），可用 RS-SAM3-p4/train_unfreeze.py 复现",
        "source": "Personal-Project/RS-SAM3-p4",
        "class": "VPT_MFNetDecoder_Unfreeze",
        "module": "train_unfreeze",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True,
                   "dropout": 0.1, "unfreeze_layers": 16},
        "use_dsm": True,
        "eval_oa": 86.02, "eval_miou": 72.58, "eval_mf1": None,
        "note": "227M params. Worse than D1. More unfreezing = more overfitting."
    },
    {
        "name": "Plan4 Full SGD (456M, MFNet recipe)",
        "phase": "Plan4 Full",
        "ckpt": "/root/autodl-tmp/archives/plan4.tar.gz",
        "archived_note": "权重已归档至 plan4.tar.gz，原始 .pt 已删除。456M all trainable, SGD+momentum+warmup. Overfit at epoch 6.",
        "source": "Personal-Project/RS-SAM3-p4",
        "class": "SAM3FullTrain",
        "module": "train_full",
        "kwargs": {"num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.37, "eval_miou": 72.87, "eval_mf1": 83.99,
        "eval_per_class_recall": {"road": 88.5, "building": 91.9, "grass": 79.6, "tree": 82.8, "car": 73.8},
        "eval_per_class_oa": {"road": 88.5, "building": 91.9, "grass": 79.6, "tree": 82.8, "car": 73.8},
        "note": "456M all trainable, SGD+momentum+warmup. Overfit at epoch 6. eval_universal already reports OA=Recall."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan5 — Boundary / Object auxiliary supervision
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan5 Boundary/Object Aux (best evaluated)",
        "phase": "Plan5",
        "ckpt": "/root/autodl-tmp/archives/plan5.tar.gz",
        "archived_note": "权重已归档至 plan5.tar.gz，原始 .pt 已删除。Boundary/object auxiliary route.",
        "source": "Personal-Project/RS-SAM3-p5",
        "class": "VPT_MFNetDecoder",
        "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.34,
        "eval_miou": 73.36,
        "eval_mf1": None,
        "protocol": "256² sliding window, stride=128",
        "note": "Boundary/object auxiliary route. Module/class info fixed 2026-05-11."
    },
    {
        "name": "Plan5 Boundary/Object Aux (rerun)",
        "phase": "Plan5",
        "ckpt": None,
        "deleted_note": "权重已删除（被 153225 超越），可用 RS-SAM3-p5/train.py 复现",
        "source": "Personal-Project/RS-SAM3-p5",
        "class": "VPT_MFNetDecoder",
        "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.38,
        "eval_miou": 73.27,
        "eval_mf1": None,
        "protocol": "256² sliding window, stride=128",
        "note": "Plan5 rerun. Module/class info fixed 2026-05-11."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan6 — MFNet-style in-ViT RGB/DSM MMAdapter (RS-SAM3-p6)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan6 Phase1 MMAdapter DSM-lite",
        "phase": "Plan6 Phase 1",
        "ckpt": None,
        "deleted_note": "权重已删除（被 phase1_5 超越），可用 Personal-Project/RS-SAM3-p6/phase1_mm_adapter/train_phase1.py 复现",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p6/phase1_mm_adapter",
        "class": "Plan6MMAdapterMFNet",
        "module": "model_phase1",
        "kwargs": {
            "adapter_bottleneck": 32,
            "num_classes": 5,
            "dropout": 0.1,
            "dsm_attn_mode": "adapter",
            "checkpoint_attn": False,
        },
        "use_dsm": True,
        "eval_oa": 87.06,
        "eval_miou": 74.88,
        "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 76.21,
            "building": 86.34,
            "grass": 60.51,
            "tree": 76.43,
            "car": 74.89,
        },
        "eval_per_class_recall": {
            "road": 90.67,
            "building": 91.76,
            "grass": 74.03,
            "tree": 85.00,
            "car": 82.33,
        },
        "eval_per_class_oa": {
            "road": 92.52,
            "building": 95.61,
            "grass": 92.29,
            "tree": 93.92,
            "car": 99.70,
        },
        "protocol": "256² sliding window, stride=128",
        "note": "First evaluated Plan6 result. Crop best 76.32% @ E008, final 256² mIoU 74.81%. Strongest evaluated checkpoint so far."
    },
    {
        "name": "Plan6 Phase1 full DSM self-attention (GLOBAL BEST)",
        "phase": "Plan6 Phase 1",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p6/phase1_mm_adapter",
        "class": "Plan6MMAdapterMFNet",
        "module": "model_phase1",
        "kwargs": {
            "adapter_bottleneck": 32,
            "num_classes": 5,
            "dropout": 0.1,
            "dsm_attn_mode": "full",
            "checkpoint_attn": True,
        },
        "use_dsm": True,
        "eval_oa": 87.27,
        "eval_miou": 76.57,
        "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.01,
            "building": 86.88,
            "grass": 63.53,
            "tree": 77.67,
            "car": 77.80,
        },
        "eval_per_class_recall": {
            "road": 90.30,
            "building": 91.52,
            "grass": 77.79,
            "tree": 85.44,
            "car": 87.66,
        },
        "eval_per_class_oa": {
            "road": 92.68,
            "building": 95.94,
            "grass": 92.33,
            "tree": 93.88,
            "car": 99.72,
        },
        "train_best_miou": 76.60,
        "train_best_epoch": 10,
        "protocol": "256² sliding window, stride=128",
        "note": "Full DSM self-attention + checkpoint, batch=2. Crop best 76.86% @ E010. Final 256² mIoU 76.57% (old per-patch argmax eval). Soft-logit eval: 77.38%."
    },
    {
        "name": "Plan6 P2-D1: Multi-scale TTA on Phase1 best (scales=1.0+0.75)",
        "phase": "Plan6 Phase 2 D1",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt",
        "init_from": "same as Plan6 Phase1 (eval only, no training)",
        "lineage_type": "eval_only",
        "comparison_role": "Zero-training multi-scale on Plan6 Phase1; test if patch-grid bias is prompt-independent",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p6/phase1_mm_adapter",
        "class": "Plan6MMAdapterMFNet",
        "module": "eval_p2d1_ms",
        "kwargs": {"scales": [1.0, 0.75]},
        "use_dsm": True,
        "eval_oa": 87.83, "eval_miou": 77.35, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.36, "building": 88.93, "grass": 64.84, "tree": 77.74, "car": 77.87,
        },
        "eval_per_class_recall": {
            "road": 90.65, "building": 92.10, "grass": 78.77, "tree": 85.85, "car": 90.62,
        },
        "train_best_miou": None, "train_best_epoch": None,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation, multi-scale avg",
        "note": "Multi-scale on Plan6: mIoU=77.35 vs soft-logit single-scale 77.38 (-0.03pp). Same conclusion as Plan7-D1: multi-scale TTA does NOT help SAM3+ViTDet on ISPRS. Soft-logit eval (+0.81pp over old eval) is the real gain."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan6 Phase 4 — MFNet 严格对照消融 (RS-SAM3-p6/phase4_mfnet_ablation)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Phase4 F0: shared encoder + LoRA + SEFusion (MFNet on SAM3)",
        "phase": "Plan6 Phase 4 F0",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260518_213923/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "from scratch",
        "comparison_role": "MFNet-equivalent baseline on SAM3: shared encoder + LoRA + SEFusion late fusion",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3",
        "class": "MFNetSAM3", "module": "model_f0",
        "kwargs": {"lora_rank": 8, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 88.23, "eval_miou": 77.34, "eval_mf1": None,
        "eval_per_class_iou": {"road": 78.7, "building": 89.9, "grass": 64.0, "tree": 77.9, "car": 76.2},
        "eval_per_class_recall": {"road": 89.5, "building": 95.0, "grass": 76.1, "tree": 87.2, "car": 86.8},
        "train_best_miou": 76.74, "train_best_epoch": 3,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation",
        "note": "Shared encoder + LoRA(rank=8) + SEFusion + MFNetDecoder. Soft-logit 77.34% vs Plan7-A soft-logit 77.55% (-0.21pp). LoRA+simple arch ≈ frozen+complex arch. Key contribution: first fair controlled comparison of MFNet vs adapter on SAM3."
    },
    # Phase4 F0 scheduler/optimizer ablation — both negative
    {
        "name": "Phase4 F0 (AdamW + MultiStepLR, 50ep) — FAILED",
        "phase": "Plan6 Phase 4 F0 scheduler ablation",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260607_084046/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "formal_ablation",
        "comparison_role": "MultiStepLR vs CosineAnnealing on AdamW. Baseline: F0 CosineAnnealing 77.34%.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3",
        "class": "MFNetSAM3", "module": "model_f0",
        "kwargs": {"lora_rank": 8, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": 76.57,
        "train_best_miou": 76.57, "train_best_epoch": 7,
        "protocol": "256² sliding window, crop validation, AdamW + MultiStepLR([25,35,45], gamma=0.1), 35ep (stopped early)",
        "note": "❌ AdamW+MultiStepLR: E7 peak 76.57%, then plateau→degrade. -0.77pp vs CosineAnnealing 77.34%. Cause: AdamW adaptive state conflicts with step-wise lr drops; constant high lr for 25 epochs drives AdamW into sharp minimum."
    },
    {
        "name": "Phase4 F0 (SGD + MultiStepLR, 50ep) — FAILED",
        "phase": "Plan6 Phase 4 F0 scheduler ablation",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260607_174443/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "formal_ablation",
        "comparison_role": "SGD+MultiStepLR vs AdamW+CosineAnnealing. Tests whether optimizer alone explains MultiStepLR failure.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3",
        "class": "MFNetSAM3", "module": "model_f0",
        "kwargs": {"lora_rank": 8, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": 76.55,
        "train_best_miou": 76.55, "train_best_epoch": 3,
        "protocol": "256² sliding window, crop validation, SGD(lr=0.01,momentum=0.9)+MultiStepLR([25,35,45]), 24ep (stopped early)",
        "note": "❌ SGD+MultiStepLR: E3 peak 76.55%, then plateau. -0.79pp vs CosineAnnealing. SGD did NOT rescue MultiStepLR — Vaihingen only 12 tiles, model saturates in 3 epochs regardless of optimizer. Root cause is Vaihingen too small for long MultiStepLR schedule, not optimizer choice."
    },
    {
        "name": "Phase4 F1: in-ViT MMAdapter + LoRA",
        "phase": "Plan6 Phase 4 F1",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_lora_vaihingen_20260519_124902/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "from scratch",
        "comparison_role": "in-ViT adapter + LoRA vs F0 (shared+LoRA). Isolates fusion strategy contribution.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion",
        "class": "Phase4AdapterModel", "module": "model_f1",
        "kwargs": {"use_lora": True, "use_prompt": False, "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 88.12, "eval_miou": 77.29, "eval_mf1": None,
        "eval_per_class_iou": {"road": 78.4, "building": 90.3, "grass": 63.3, "tree": 77.7, "car": 76.8},
        "eval_per_class_recall": {"road": 91.6, "building": 94.3, "grass": 74.8, "tree": 86.3, "car": 88.4},
        "train_best_miou": 76.83, "train_best_epoch": 7,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation",
        "note": "F1-F0 = -0.11pp (soft-logit). in-ViT adapter ≈ shared encoder + SEFusion when both use LoRA. Fusion strategy matters less when backbone is LoRA-tuned."
    },
    {
        "name": "Phase4 F2: in-ViT MMAdapter frozen (no LoRA)",
        "phase": "Plan6 Phase 4 F2",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_frozen_vaihingen_20260519_155635/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_lora_vaihingen_20260519_124902/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Frozen vs LoRA. Isolates parameter strategy contribution.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion",
        "class": "Phase4AdapterModel", "module": "model_f1",
        "kwargs": {"use_lora": False, "use_prompt": False, "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 86.63, "eval_miou": 75.51, "eval_mf1": None,
        "eval_per_class_iou": {"road": 75.5, "building": 87.0, "grass": 61.2, "tree": 77.2, "car": 76.7},
        "eval_per_class_recall": {"road": 91.0, "building": 90.9, "grass": 74.6, "tree": 85.0, "car": 89.6},
        "train_best_miou": 74.76, "train_best_epoch": 7,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation",
        "note": "F2-F1 = -1.49pp. LoRA contributes ~1.5pp, the largest single factor. Removing LoRA causes significant degradation that 8-epoch adapter-only training cannot fully recover."
    },
    {
        "name": "Phase4 F3: in-ViT MMAdapter frozen + DSM edge/slope prompt",
        "phase": "Plan6 Phase 4 F3",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_prompt_vaihingen_20260519_191703/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_frozen_vaihingen_20260519_155635/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "+prompt vs frozen. Isolates structural prior contribution.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f1_fusion",
        "class": "Phase4AdapterModel", "module": "model_f1",
        "kwargs": {"use_lora": False, "use_prompt": True, "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 87.28, "eval_miou": 76.52, "eval_mf1": None,
        "eval_per_class_iou": {"road": 76.3, "building": 87.7, "grass": 62.8, "tree": 78.1, "car": 77.8},
        "eval_per_class_recall": {"road": 90.8, "building": 91.8, "grass": 75.0, "tree": 86.5, "car": 89.8},
        "train_best_miou": 75.93, "train_best_epoch": 8,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation",
        "note": "F3-F2 = +0.65pp. Prompt contributes consistently (+0.57pp in Plan7-A, +0.65pp here). 76.52% mIoU close to Plan6 Phase1 (76.57%), validating architecture reproducibility."
    },

    # Phase 4 third supplement — MMLoRA alignment
    {
        "name": "Phase4 F0'+L2: Frozen SAM3 + 4xSEFusion + LoRA(attn-only)",
        "phase": "Plan6 Phase 4 F0'+L2",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_l2_vaihingen_20260520_192526/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "LoRA attn-only vs attn+MLP; matches MFNet MMLoRA injection position",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline",
        "class": "FrozenSAM3DFMLoRAv2", "module": "model_f0p_l2",
        "kwargs": {"num_classes": 5, "dropout": 0.1, "lora_rank": 8},
        "use_dsm": True,
        "eval_oa": 87.57, "eval_miou": 77.01, "eval_mf1": None,
        "eval_per_class_iou": {"road": 76.7, "building": 87.8, "grass": 64.1, "tree": 78.3, "car": 78.1},
        "eval_per_class_recall": {"road": 91.3, "building": 91.2, "grass": 76.1, "tree": 87.0, "car": 88.9},
        "train_best_miou": 76.56, "train_best_epoch": 7,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "LoRA attn-only: -0.23pp vs F0'+L (attn+MLP). Removing MLP LoRA costs slightly but closes gap to MFNet MMLoRA injection position."
    },
    {
        "name": "Phase4 F0'+M: SAM3 + MMLoRA (attn LoRA + per-block λ mixing, 4 global blocks)",
        "phase": "Plan6 Phase 4 F0'+M",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_m_vaihingen_20260520_223449/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_l2_vaihingen_20260520_192526/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "MMLoRA dual-branch λ mixing vs plain LoRA; tests encoder-internal cross-modal fusion",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline",
        "class": "FrozenSAM3MMLoRA", "module": "model_f0p_m",
        "kwargs": {"num_classes": 5, "dropout": 0.1, "lora_rank": 8, "mixing_rank": 8},
        "use_dsm": True,
        "eval_oa": 87.64, "eval_miou": 76.89, "eval_mf1": None,
        "eval_per_class_iou": {"road": 77.0, "building": 88.6, "grass": 63.9, "tree": 77.7, "car": 77.2},
        "eval_per_class_recall": {"road": 91.5, "building": 92.7, "grass": 76.4, "tree": 85.1, "car": 88.3},
        "train_best_miou": 76.45, "train_best_epoch": 6,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "MMLoRA λ mixing: -0.35pp vs F0'+L. Encoder-internal dual-branch fusion is REDUNDANT when LoRA already active on SAM3. Direction closed."
    },

    # ═══════════════════════════════════════════════════════════
    # Phase 4 unfreeze experiments — U1/U2/U3
    {
        "name": "Phase4 U1: Frozen SAM3 + 4xSEFusion, unfreeze blocks 24-31 attn",
        "phase": "Plan6 Phase 4 U1",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_u24_31_vaihingen_20260525_211145/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Clean test of MMA deep-unfreeze hypothesis on SAM3 (fixed windows, normal lr)",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline/unfreeze",
        "class": "FrozenSAM3DFM", "module": "model_f0p",
        "kwargs": {"num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 87.09, "eval_miou": 76.08, "eval_mf1": None,
        "eval_per_class_iou": {"road": 76.1, "building": 87.9, "grass": 62.6, "tree": 77.1, "car": 76.6},
        "eval_per_class_recall": {"road": 91.1, "building": 92.1, "grass": 75.7, "tree": 84.6, "car": 87.9},
        "train_best_miou": 75.49, "train_best_epoch": 10,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "Unfreeze blocks 24-31 attn only, fixed windows, lr=1e-5. +0.20pp crop vs frozen. 256²: 76.08%. MMA deep-unfreeze hypothesis marginally supported on SAM3."
    },
    {
        "name": "Phase4 U2: Frozen SAM3 + 4xSEFusion, unfreeze blocks 24-31 attn+MLP",
        "phase": "Plan6 Phase 4 U2",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_u24_31_mlp_vaihingen_20260525_232410/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Unfreeze attn+MLP vs attn-only (U1)",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline/unfreeze",
        "class": "FrozenSAM3DFM", "module": "model_f0p",
        "kwargs": {"num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 87.35, "eval_miou": 76.24, "eval_mf1": None,
        "eval_per_class_iou": {"road": 76.5, "building": 88.5, "grass": 62.1, "tree": 77.7, "car": 76.5},
        "eval_per_class_recall": {"road": 91.2, "building": 92.4, "grass": 72.1, "tree": 87.7, "car": 88.5},
        "train_best_miou": 75.55, "train_best_epoch": 4,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "Unfreeze attn+MLP: similar to U1. Adding MLP to unfreeze doesn't help more."
    },
    {
        "name": "Phase4 U3: Frozen SAM3 + 4xSEFusion, unfreeze blocks 28-31 attn (Phase 1.6 repl)",
        "phase": "Plan6 Phase 4 U3",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_u28_31_vaihingen_20260526_005659/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Phase 1.6 replication with fixed windows + normal lr",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline/unfreeze",
        "class": "FrozenSAM3DFM", "module": "model_f0p",
        "kwargs": {"num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.96, "eval_miou": 75.87, "eval_mf1": None,
        "eval_per_class_iou": {"road": 76.1, "building": 87.8, "grass": 61.9, "tree": 77.0, "car": 76.6},
        "eval_per_class_recall": {"road": 91.3, "building": 92.1, "grass": 75.0, "tree": 84.5, "car": 87.2},
        "train_best_miou": 75.23, "train_best_epoch": 7,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "Phase 1.6 fixed-window replication: -0.06pp crop. Original Phase 1.6 degradation (-4pp) was caused by online crops + ultra-low lr, not unfreezing per se. But unfreezing still doesn't help."
    },

    # Plan9 — LoRA + DSM edge/slope input fusion (RS-SAM3-p9)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan9 P9-A: F0'+L + 3ch DSM (DSM+edge+slope) input enrichment",
        "phase": "Plan9 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan9_p9A_vaihingen_20260526_143908/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "LoRA + input-level edge/slope prompt vs LoRA-only (F0'+L). Tests if prompt works without adapter gate.",
        "source": "Personal-Project/RS-SAM3-p9/phase_a_lora_prompt",
        "class": "Plan9ModelA", "module": "train_p9",
        "kwargs": {"num_classes": 5, "lora_rank": 8},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": None, "eval_mf1": None,
        "train_best_miou": 76.45, "train_best_epoch": 8,
        "protocol": "crop validation only (256² eval not run — negative result confirmed)",
        "note": "3ch DSM (DSM+edge+slope) directly into ViTDet. Crop best 76.45% vs F0'+L 76.81% (-0.36pp). Edge/slope prompt works ONLY with adapter gate (Plan7-A), not input-level enrichment. P9-B skipped. Plan9 closed."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan10 — Potsdam 跨数据集验证 (RS-SAM3-p10)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan10 P10-A: Plan7-A on Potsdam (from scratch, 20 epoch)",
        "phase": "Plan10 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_potsdam_20260526_210230/best_model.pt",
        "init_from": "from scratch (seed default)",
        "lineage_type": "from scratch",
        "comparison_role": "Potsdam cross-dataset: adapter+prompt architecture on larger dataset",
        "source": "Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt",
        "class": "Plan7PromptMFNet", "module": "model",
        "kwargs": {"num_classes": 5, "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 90.98, "eval_miou": 79.95, "eval_mf1": None,
        "eval_per_class_iou": {"road": 80.2, "building": 92.4, "grass": 69.9, "tree": 73.4, "car": 83.8},
        "train_best_miou": 81.83, "train_best_epoch": 18,
        "protocol": "256² sliding window, soft-logit, Potsdam dataset",
        "note": "Potsdam 20 epoch from scratch. 79.95% 256² mIoU. Adapter route performs well but slightly below LoRA routes on larger dataset."
    },
    {
        "name": "Plan10 P10-B: F0 on Potsdam (shared encoder + LoRA + 1xSEF, 20 epoch)",
        "phase": "Plan10 Phase B",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0_potsdam_20260527_092853/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "from scratch",
        "comparison_role": "Potsdam cross-dataset: LoRA+shared encoder on larger dataset",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3",
        "class": "MFNetSAM3", "module": "model_f0",
        "kwargs": {"num_classes": 5, "lora_rank": 8},
        "use_dsm": True,
        "eval_oa": 91.62, "eval_miou": 80.91, "eval_mf1": None,
        "eval_per_class_iou": {"road": 82.5, "building": 90.2, "grass": 72.1, "tree": 75.0, "car": 84.7},
        "train_best_miou": 82.12, "train_best_epoch": 14,
        "protocol": "256² sliding window, soft-logit, Potsdam dataset",
        "note": "Potsdam 20 epoch from scratch. 80.91% 256² mIoU. LoRA on shared encoder competitive with F0'+L. All 3 routes within 1pp on Potsdam."
    },
    {
        "name": "Plan10 P10-C: F0'+L on Potsdam (frozen+LoRA+4xSEF, 20 epoch)",
        "phase": "Plan10 Phase C",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_potsdam_20260528_105807/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_potsdam_20260528_035103/best_model.pt",
        "lineage_type": "continuation",
        "comparison_role": "Potsdam cross-dataset: frozen+LoRA+4xSEF on larger dataset",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline",
        "class": "FrozenSAM3DFMLoRA", "module": "model_f0p_lora",
        "kwargs": {"num_classes": 5, "lora_rank": 8},
        "use_dsm": True,
        "eval_oa": 91.67, "eval_miou": 80.95, "eval_mf1": None,
        "eval_per_class_iou": {"road": 82.0, "building": 90.8, "grass": 72.2, "tree": 75.3, "car": 84.4},
        "train_best_miou": 82.17, "train_best_epoch": 19,
        "protocol": "256² sliding window, soft-logit, Potsdam dataset",
        "note": "Potsdam best (80.95%). LoRA on frozen+4xSEF. All 3 routes within 1pp — architecture choice not critical cross-dataset."
    },

    # Phase 4 supplementary — clean frozen SAM3 baseline
    {
        "name": "Phase4 F0': Frozen SAM3 + 4xSEFusion (DFM) + MFNetDecoder",
        "phase": "Plan6 Phase 4 F0'",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt",
        "init_from": "from scratch (seed=42)",
        "lineage_type": "from scratch",
        "comparison_role": "Strict MFNet \"Without Adapter\" equivalent on SAM3. Clean frozen baseline.",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline",
        "class": "FrozenSAM3DFM", "module": "model_f0p",
        "kwargs": {"num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
        "eval_oa": 86.57, "eval_miou": 75.29, "eval_mf1": None,
        "eval_per_class_iou": {"road": 75.5, "building": 86.9, "grass": 60.8, "tree": 77.0, "car": 76.3},
        "eval_per_class_recall": {"road": 90.2, "building": 91.8, "grass": 73.5, "tree": 85.4, "car": 87.1},
        "train_best_miou": 74.53, "train_best_epoch": 12,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "Clean frozen SAM3 baseline (no LoRA, no adapter). 75.29% mIoU vs MFNet Without Adapter 75.11% (+0.18pp). SAM3 frozen ≈ SAM1 frozen. LoRA in SAM3 adds +1.95pp (77.24%), vs SAM1's +6.95pp (82.06%). SAM3 is more rigid to fine-tuning."
    },
    {
        "name": "Phase4 F0'+L: Frozen SAM3 + 4xSEFusion + LoRA (rank=8)",
        "phase": "Plan6 Phase 4 F0'+L",
        "ckpt": "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan6_phase4_f0p_frozen_vaihingen_20260519_222933/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "LoRA's net contribution on clean frozen SAM3 baseline (+1.95pp)",
        "source": "Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline",
        "class": "FrozenSAM3DFMLoRA", "module": "model_f0p_lora",
        "kwargs": {"num_classes": 5, "dropout": 0.1, "lora_rank": 8},
        "use_dsm": True,
        "eval_oa": 87.89, "eval_miou": 77.24, "eval_mf1": None,
        "eval_per_class_iou": {"road": 77.5, "building": 88.3, "grass": 64.7, "tree": 78.3, "car": 77.4},
        "eval_per_class_recall": {"road": 90.1, "building": 93.0, "grass": 76.5, "tree": 87.3, "car": 86.7},
        "train_best_miou": 76.81, "train_best_epoch": 1,
        "protocol": "256² sliding window, soft-logit accumulation",
        "note": "LoRA on clean frozen SAM3 baseline: +1.95pp (75.29→77.24). LoRA peaks at E1 (76.81%), then overfits. 4xSEFusion slightly underperforms 1xSEFusion (F0: 77.34%) in LoRA setting."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan7 — 多模态结构先验注入 (RS-SAM3-p7)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan7-A: MMAdapter + DSM edge/slope prompt",
        "phase": "Plan7 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_225309/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_215044/best_model.pt",
        "parent_of_parent": "/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Plan6 Phase1 full -> add DSM edge/slope prompt (via 215044 short run, 2-epoch warmup)",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt",
        "class": "Plan7PromptMFNet",
        "module": "model",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1,
                   "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 87.75, "eval_miou": 77.14, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.61,
            "building": 87.87,
            "grass": 64.62,
            "tree": 78.16,
            "car": 77.44,
        },
        "eval_per_class_recall": {
            "road": 88.52,
            "building": 93.00,
            "grass": 79.70,
            "tree": 86.06,
            "car": 91.59,
        },
        "train_best_miou": 76.87,
        "train_best_epoch": 3,
        "protocol": "256² sliding window, stride=128, global confusion matrix (soft-logit accumulation)",
        "note": "Init from 215044 (2-epoch Plan7-A short run, dir deleted), parent is Plan6 Phase1. Global eval 77.14 is soft-logit accumulation (per-patch argmax = 76.27). D1 single-scale soft-logit eval = 77.55 (different eval script, yields ~0.41pp higher than original eval). Current Plan7 best; B1/B3 did not improve over edge+slope."
    },
    {
        "name": "Plan7-B1: DSM slope-only prompt",
        "phase": "Plan7 Phase B1",
        "ckpt": "/root/autodl-tmp/archives/plan7_archivable.tar.gz",
        "archived_note": "权重已归档至 plan7_archivable.tar.gz，原始 .pt 已删除。SAM-HQ residual correction on top of Plan7-A. 77.05 vs 77.14, no gain.",
        "init_from": "/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Prompt-shape ablation vs Plan7-A; only prompt channels change",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p7/phase_b_prompt_ablation/b1_slope_only",
        "class": "Plan7PromptMFNet",
        "module": "model",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1,
                   "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 87.26, "eval_miou": 76.49, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.05,
            "building": 86.50,
            "grass": 63.38,
            "tree": 77.98,
            "car": 77.54,
        },
        "eval_per_class_recall": {
            "road": 90.83,
            "building": 91.70,
            "grass": 77.12,
            "tree": 85.14,
            "car": 86.44,
        },
        "train_best_miou": 76.77,
        "train_best_epoch": 6,
        "protocol": "256² sliding window, stride=128",
        "note": "Formal B-stage ablation. Slope-only did not beat Plan7-A edge+slope: 76.49 vs 77.14 mIoU."
    },
    {
        "name": "Plan7-B3: DSM slope/edge/curvature prompt",
        "phase": "Plan7 Phase B3",
        "ckpt": None,
        "deleted_note": "权重已删除（与 B1 同分，冗余），可用 Personal-Project/RS-SAM3-p7/phase_b_prompt_ablation/b3 复现",
        "init_from": "/root/autodl-tmp/runs/plan6_phase1_mm_adapter_vaihingen_20260509_202720/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Prompt-shape ablation vs Plan7-A; only prompt channels change",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p7/phase_b_prompt_ablation/b3_slope_edge_curvature",
        "class": "Plan7PromptMFNet",
        "module": "model",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1,
                   "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 87.33, "eval_miou": 76.49, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.39,
            "building": 87.77,
            "grass": 63.58,
            "tree": 76.95,
            "car": 76.74,
        },
        "eval_per_class_recall": {
            "road": 89.50,
            "building": 92.76,
            "grass": 80.64,
            "tree": 83.00,
            "car": 90.66,
        },
        "train_best_miou": 76.62,
        "train_best_epoch": 1,
        "protocol": "256² sliding window, stride=128",
        "note": "Formal B-stage ablation. Curvature did not improve over Plan7-A edge+slope: 76.49 vs 77.14 mIoU."
    },
    {
        "name": "Plan7-C3: SAM-HQ residual boundary correction (GlobalLocalFusion + ResidualHead)",
        "phase": "Plan7 Phase C3",
        "ckpt": "/root/autodl-tmp/archives/plan7_archivable.tar.gz",
        "archived_note": "权重已归档至 plan7_archivable.tar.gz，原始 .pt 已删除。SAM-HQ residual correction on top of Plan7-A. 77.05 vs 77.14, no gain.",
        "init_from": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_215044/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "Plan7-A best -> add HQ-Fusion + residual correction head; test SAM-HQ paradigm on SAM3",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p7/phase_c_residual_boundary/c3_full",
        "class": "Plan7C3MFNet",
        "module": "model_c3",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1,
                   "dsm_attn_mode": "full", "checkpoint_attn": True},
        "use_dsm": True,
        "eval_oa": 87.67, "eval_miou": 77.05, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.77,
            "building": 87.76,
            "grass": 63.83,
            "tree": 78.02,
            "car": 77.84,
        },
        "eval_per_class_recall": {
            "road": 90.00,
            "building": 93.02,
            "grass": 77.95,
            "tree": 85.45,
            "car": 88.29,
        },
        "train_best_miou": 77.15,
        "train_best_epoch": 2,
        "protocol": "256² sliding window, stride=128",
        "note": "SAM-HQ residual correction on top of Plan7-A. mIoU=77.05 vs Plan7-A 77.14 (-0.09pp, noise-level). Grass +0.83pp, tree +0.32pp, car +0.34pp improved slightly; road -0.42pp, building -0.54pp regressed slightly. Overall: SAM-HQ paradigm did NOT transfer to SAM3+MFNetDecoder. C4 (prompt dropout) skipped."
    },
    {
        "name": "Plan7-D1: Multi-scale TTA on Plan7-A best (soft-logit eval, scales=1.0+0.75)",
        "phase": "Plan7 Phase D1",
        "ckpt": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_215044/best_model.pt",
        "init_from": "same as Plan7-A (eval only, no training)",
        "lineage_type": "eval_only",
        "comparison_role": "Zero-training multi-scale inference on top of Plan7-A best",
        "source": "Personal-Project/Personal-Project/RS-SAM3-p7/phase_d_multiscale_spatial/d1_multiscale_eval",
        "class": "Plan7PromptMFNet",
        "module": "eval_ms",
        "kwargs": {"scales": [1.0, 0.75]},
        "use_dsm": True,
        "eval_oa": 88.11, "eval_miou": 77.33, "eval_mf1": None,
        "eval_per_class_iou": {
            "road": 77.67,
            "building": 89.47,
            "grass": 65.20,
            "tree": 78.29,
            "car": 76.03,
        },
        "eval_per_class_recall": {
            "road": 89.06,
            "building": 93.63,
            "grass": 79.51,
            "tree": 86.29,
            "car": 92.14,
        },
        "train_best_miou": None,
        "train_best_epoch": None,
        "protocol": "256² sliding window, stride=128, soft-logit accumulation, multi-scale avg",
        "note": "D1 multi-scale (1.0+0.75) on Plan7-A: mIoU=77.33 vs soft-logit single-scale 77.55 (-0.22pp). Car regressed -0.81pp. Multi-scale does NOT help SAM3+ViTDet on ISPRS at 9cm GSD. Side finding: soft-logit accumulation (+0.41pp over per-patch argmax) is the real gain. All future eval should use soft-logit."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan6 Phase 3 — 消融实验 (Personal-Project/RS-SAM3-p6/phase3_ablation)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Phase3-A0: SAM3 raw ViT + Conv2d head (anchor)",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part1.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part1.tar.gz，原始 .pt 已删除。Ablation anchor.",
        "eval_oa": 70.77, "eval_miou": 53.10,
        "eval_per_class_iou": {"road": 56.2, "building": 63.1, "grass": 29.0, "tree": 65.3, "car": 51.8},
        "train_best_miou": 56.42,
        "protocol": "256² sliding window, global confusion matrix",
        "note": "Ablation anchor. SAM3 frozen + Conv2d head only, no adapter/DSM/decoder."
    },
    {
        "name": "Phase3-A1: A0 + DSM late fusion + SimpleDecoder",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part1.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part1.tar.gz，原始 .pt 已删除。Ablation anchor.",
        "eval_oa": 80.64, "eval_miou": 65.86,
        "eval_per_class_iou": {"road": 67.7, "building": 75.8, "grass": 50.2, "tree": 72.3, "car": 63.3},
        "train_best_miou": 69.41,
        "protocol": "256² sliding window, global confusion matrix",
        "note": "Ablation: +DSM via SEFusion late fusion + Pyramid4Scale + SimpleDecoder. +12.99pp over A0."
    },
    {
        "name": "Phase3-A3: A1 + in-ViT MMAdapter full + SimpleDecoder",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part1.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part1.tar.gz，原始 .pt 已删除。Ablation anchor.",
        "eval_oa": 86.00, "eval_miou": 72.88,
        "eval_per_class_iou": {"road": 75.0, "building": 85.4, "grass": 61.4, "tree": 76.2, "car": 66.3},
        "train_best_miou": 76.16,
        "protocol": "256² sliding window, global confusion matrix",
        "note": "Ablation: in-ViT MMAdapter replaces late fusion. +6.75pp over A1. MMAdapter is the core component."
    },
    {
        "name": "Phase3-C0: A3 + fixed windows (vs online crops)",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part1.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part1.tar.gz，原始 .pt 已删除。Ablation anchor.",
        "train_best_miou": 76.83,
        "note": "Ablation: fixed windows. E1=76.83% then degraded. +0.67pp over A3 online crops."
    },
    {
        "name": "Phase3-D1: A3 + CE loss (vs structure_loss)",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part2.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part2.tar.gz，原始 .pt 已删除。CrossEntropyLoss. best=76.34% vs A3 structure_loss 76.16%. +0.18pp, noise-level.",
        "train_best_miou": 76.34,
        "note": "Ablation: CrossEntropyLoss. best=76.34% vs A3 structure_loss 76.16%. +0.18pp, noise-level."
    },
    {
        "name": "Phase3-B1: A3 + softmax gate (vs sigmoid)",
        "phase": "Plan6 Phase 3",
        "ckpt": "/root/autodl-tmp/archives/plan6_archivable_part2.tar.gz",
        "archived_note": "权重已归档至 plan6_archivable_part2.tar.gz，原始 .pt 已删除。CrossEntropyLoss. best=76.34% vs A3 structure_loss 76.16%. +0.18pp, noise-level.",
        "train_best_miou": 76.20,
        "note": "Ablation: softmax 2-way gate. best=76.20% vs A3 sigmoid 76.16%. +0.04pp, no difference."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan8 — 增强 In-ViT RGB↔DSM 交互 (RS-SAM3-p8)
    # Result: NEGATIVE. Both cross-attn and attn bias provide no gain over gate-only.
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Plan8-CTRL: gate-only (Plan7-A arch, trained from scratch)",
        "phase": "Plan8 CTRL",
        "ckpt": "/root/autodl-tmp/runs/plan8_ctrl_vaihingen_20260516_081446/best_model.pt",
        "eval_oa": 86.33, "eval_miou": 74.77,
        "eval_per_class_iou": {"road": 76.34, "building": 84.53, "grass": 62.47, "tree": 76.75, "car": 73.78},
        "eval_per_class_recall": {"road": 88.56, "building": 89.82, "grass": 81.82, "tree": 83.20, "car": 79.09},
        "train_best_miou": 75.19,
        "protocol": "256² sliding window, soft-logit overlap averaging",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "note": "Plan7-A architecture trained from scratch (seed=42). -2.37pp vs Plan7-A from Plan6 ckpt."
    },
    {
        "name": "Plan8-CA-A: gate + unidirectional DSM→RGB cross-attention",
        "phase": "Plan8 Chain 1 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan8_ca_a_vaihingen_20260516_131457/best_model.pt",
        "eval_oa": 86.21, "eval_miou": 74.52,
        "eval_per_class_iou": {"road": 76.13, "building": 84.80, "grass": 61.47, "tree": 76.62, "car": 73.57},
        "eval_per_class_recall": {"road": 89.22, "building": 89.88, "grass": 79.48, "tree": 83.57, "car": 78.86},
        "train_best_miou": 75.36,
        "protocol": "256² sliding window, soft-logit overlap averaging",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "comparison_role": "vs CTRL: cross-attn effect",
        "note": "-0.26pp vs CTRL. Cross-attn in frozen ViT provides no benefit. Chain 1 stopped."
    },
    {
        "name": "Plan8-AB-A: gate + DSM elevation attention bias",
        "phase": "Plan8 Chain 2 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan8_ab_a_vaihingen_20260516_144112/best_model.pt",
        "eval_oa": 86.22, "eval_miou": 74.12,
        "eval_per_class_iou": {"road": 75.67, "building": 83.05, "grass": 63.13, "tree": 77.49, "car": 71.28},
        "eval_per_class_recall": {"road": 90.14, "building": 88.50, "grass": 78.23, "tree": 85.23, "car": 75.79},
        "train_best_miou": 74.68,
        "protocol": "256² sliding window, soft-logit overlap averaging",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "comparison_role": "vs CTRL: attn bias effect",
        "note": "-0.65pp vs CTRL. DSM attn bias degrades performance. Chain 2 stopped."
    },

    # ═══════════════════════════════════════════════════════════
    # Plan11 — 植被区分优化 (RS-SAM3-p11)
    # ═══════════════════════════════════════════════════════════
    # Result: nDSM +1.95pp vs from-scratch Plan8-CTRL. veg_boundary_weight consistently negative.
    {
        "name": "P11-A: min-max + veg boundary weight (from Plan7-A init)",
        "phase": "Plan11 Phase A",
        "ckpt": "/root/autodl-tmp/runs/plan11_a_veg_boundary_loss_vaihingen_20260604_202157/best_model.pt",
        "init_from": "/root/autodl-tmp/runs/plan7_phase_a_dsm_prompt_vaihingen_20260510_225309/best_model.pt",
        "lineage_type": "formal_ablation",
        "comparison_role": "veg_boundary_weight effect on min-max DSM; baseline is Plan7-A (77.14)",
        "source": "Personal-Project/RS-SAM3-p11/phase_a_veg_boundary_loss",
        "class": "Plan7PromptMFNet", "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "veg_boundary_weight": 3.0},
        "use_dsm": True,
        "eval_oa": 87.16, "eval_miou": 76.46,
        "eval_per_class_iou": {"road": 76.84, "building": 86.42, "grass": 63.64, "tree": 77.72, "car": 77.67},
        "eval_per_class_recall": {"road": 89.90, "building": 91.58, "grass": 78.43, "tree": 84.99, "car": 86.51},
        "train_best_miou": 76.43, "train_best_epoch": 3,
        "protocol": "256² sliding window, per-patch argmax",
        "note": "veg_boundary_weight=3.0 on Plan7-A min-max DSM. -0.68pp vs Plan7-A. Vegetation boundary weight is harmful."
    },
    {
        "name": "P11-B: nDSM baseline (from scratch, seed=42)",
        "phase": "Plan11 Phase B",
        "ckpt": "/root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_20260604_225146/best_model.pt",
        "init_from": None,
        "lineage_type": "from scratch",
        "comparison_role": "nDSM effect vs Plan8-CTRL (same arch, same seed, min-max DSM, 74.77)",
        "source": "Personal-Project/RS-SAM3-p11/phase_b_ndsm",
        "class": "Plan7PromptMFNet", "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "seed": 42},
        "use_dsm": True,
        "eval_oa": 87.42, "eval_miou": 76.72,
        "eval_per_class_iou": {"road": 76.84, "building": 86.47, "grass": 63.70, "tree": 78.06, "car": 78.51},
        "eval_per_class_recall": {"road": 91.25, "building": 90.80, "grass": 77.42, "tree": 85.38, "car": 88.66},
        "train_best_miou": 76.71, "train_best_epoch": 9,
        "protocol": "256² sliding window, per-patch argmax, nDSM normalization",
        "note": "nDSM + global normalization. +1.95pp vs Plan8-CTRL (74.77). Improvement from road (+1.35pp recall) and car (+2.15pp), NOT from vegetation confusion reduction."
    },
    {
        "name": "P11-C: nDSM + veg boundary weight (from scratch, seed=42)",
        "phase": "Plan11 Phase C",
        "ckpt": "/root/autodl-tmp/runs/plan11_c_combined_vaihingen_20260605_012150/best_model.pt",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "comparison_role": "veg_boundary_weight effect on nDSM; baseline is P11-B (76.72)",
        "source": "Personal-Project/RS-SAM3-p11/phase_c_combined",
        "class": "Plan7PromptMFNet", "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "veg_boundary_weight": 3.0, "seed": 42},
        "use_dsm": True,
        "eval_oa": 87.14, "eval_miou": 76.51,
        "eval_per_class_iou": {"road": 77.11, "building": 86.83, "grass": 62.84, "tree": 77.46, "car": 78.30},
        "eval_per_class_recall": {"road": 91.06, "building": 91.28, "grass": 77.00, "tree": 84.95, "car": 87.74},
        "train_best_miou": 76.40, "train_best_epoch": 15,
        "protocol": "256² sliding window, per-patch argmax, nDSM normalization",
        "note": "veg_boundary_weight on nDSM. -0.21pp vs P11-B. veg_boundary_weight consistently harmful across both DSM normalizations."
    },
    {
        "name": "P11-D: nDSM + photometric augmentation (from scratch, seed=42)",
        "phase": "Plan11 Phase D",
        "ckpt": "/root/autodl-tmp/runs/plan11_d_aug_vaihingen_20260605_035155/best_model.pt",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "comparison_role": "augmentation effect on nDSM; baseline is P11-B (76.72)",
        "source": "Personal-Project/RS-SAM3-p11/phase_d_augmentation",
        "class": "Plan7PromptMFNet", "module": "train",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "seed": 42,
                   "aug_color_jitter": 0.2, "aug_blur_prob": 0.3},
        "use_dsm": True,
        "eval_oa": 87.30, "eval_miou": 76.56,
        "eval_per_class_iou": {"road": 77.51, "building": 87.01, "grass": 62.93, "tree": 77.61, "car": 77.72},
        "eval_per_class_recall": {"road": 90.80, "building": 92.05, "grass": 76.53, "tree": 85.33, "car": 86.78},
        "train_best_miou": 76.62, "train_best_epoch": 11,
        "protocol": "256² sliding window, per-patch argmax, nDSM normalization",
        "note": "ColorJitter + GaussianBlur on nDSM. -0.16pp vs P11-B (noise-level). Photometric augmentation does not help on 12-tile Vaihingen."
    },
    {
        "name": "P11-E: nDSM + adapter ablation (bottleneck=8, 4 global blocks only, seed=42)",
        "phase": "Plan11 Phase E",
        "ckpt": "/root/autodl-tmp/runs/plan11_e_adapter_vaihingen_20260605_062329/best_model.pt",
        "init_from": None,
        "lineage_type": "formal_ablation",
        "comparison_role": "adapter parameter count effect; baseline is P11-B (76.72, 32 blocks, bn=32, ~10.65M params)",
        "source": "Personal-Project/RS-SAM3-p11/phase_e_adapter_ablation",
        "class": "Plan7PromptMFNet", "module": "train",
        "kwargs": {"adapter_bottleneck": 8, "num_classes": 5, "seed": 42, "adapter_placement": "global"},
        "use_dsm": True,
        "eval_oa": None, "eval_miou": 72.61,
        "eval_per_class_iou": {"road": 73.53, "building": 83.51, "grass": 58.92, "tree": 74.98, "car": 72.09},
        "eval_per_class_recall": {"road": 86.82, "building": 90.87, "grass": 74.75, "tree": 83.16, "car": 82.23},
        "train_best_miou": 75.75, "train_best_epoch": 15,
        "protocol": "256² sliding window, per-patch argmax, nDSM normalization",
        "note": "Adapter reduced to 4 global blocks [7,15,23,31] with bottleneck=8 (~0.33M params). -4.11pp vs P11-B. Full 32-block injection is essential."
    },
]

# ═══════════════════════════════════════════════════════════
# MFNet Paper Baselines (reference, not our checkpoints)
# ═══════════════════════════════════════════════════════════
MFNET_BASELINES = {
    # NOTE: MFNet paper "per-class OA" = TP/(TP+FN) = per-class Recall.
    # These values are comparable to our eval_per_class_recall, NOT eval_per_class_oa.
    "MFNet Frozen (SAM1, no adapter)": {
        "oa": 88.01, "miou": 75.11, "mf1": 85.34,
        "per_class_oa": {"road": 89.51, "building": 94.64, "grass": 71.71, "tree": 89.47, "car": 76.83},
        "note": "SAM1 ViT-L frozen + DFM + UNetFormer. Closest baseline to ours."
    },
    "MFNet RGB+Adapter (SAM1, standard)": {
        "oa": 92.02, "miou": 83.69, "mf1": 90.94,
        "per_class_oa": {"road": 92.59, "building": 96.29, "grass": 80.15, "tree": 93.09, "car": 89.08},
    },
    "MFNet Best (SAM1, MMAdapter)": {
        "oa": 92.93, "miou": 84.72, "mf1": 91.51,
        "per_class_oa": {"road": 93.39, "building": 98.84, "grass": 81.16, "tree": 93.17, "car": 89.23},
    },

    # ── Potsdam (our reproduction, not paper values) ──
    # Protocol: MFNet native eval — 256² sliding window, stride=128, soft-logit accumulation,
    # per-tile min-max DSM normalization, NO edge trimming, eroded labels.
    # ⚠️ Differs from our standard 256² protocol which trims min(16, patch//4) pixels from each patch edge.
    "MFNet Potsdam (our repro, SAM1 LoRA+Decoder, SEG+BDY+OBJ)": {
        "dataset": "potsdam",
        "oa": 90.69, "miou": 85.14, "kappa": 0.8770, "mf1": 91.79,
        "per_class_iou": {"road": 86.11, "building": 93.98, "grass": 75.58, "tree": 76.87, "car": 92.82},
        "per_class_f1":  {"road": 92.48, "building": 97.00, "grass": 86.28, "tree": 86.93, "car": 96.25},
        "best_epoch": 42,
        "ckpt": "/root/autodl-tmp/runs/mfnet_potsdam_20260606_152557/UNetformer_best.pth",
        "source": "Reference-Project/MFNet/train.py",
        "note": "Reproduced with Reference-Project/MFNet/train.py, 50 epochs. SAM1 ViT-L frozen + LoRA (2.75M) + UNetFormer decoder (6.22M) = 8.97M trainable. SEG+BDY+OBJ loss with 10-epoch structure warmup. Potsdam 18 train / 6 test tiles. 5-class mIoU excludes clutter (clutter=46.49%)."
    },
}
