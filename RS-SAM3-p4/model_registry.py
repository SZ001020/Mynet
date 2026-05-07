"""
模型注册表：记录每个实验版本 → 对应代码和加载参数。

每个入口包含:
- ckpt_path: checkpoint 路径
- source_dir: 代码所在目录
- model_class: 类名
- init_kwargs: 构造参数
- description: 实验描述

新增实验时在此文件追加入口即可。
"""

REGISTRY = [
    # ── Plan3 Route A (RS-SAM3-p3) ────────────────────────
    {
        "name": "Route A RGB (VPT + simple UNet)",
        "ckpt": "/root/autodl-tmp/runs/plan3_adapter_20260501_224049/best_model.pt",
        "source": "RS-SAM3-p3",
        "class": "AdapterSAM3UNet",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5},
        "use_dsm": False,
    },

    # ── Plan3 Route A+DSM (RS-SAM3-p3r) ───────────────────
    {
        "name": "Route A+DSM cross-attn (VPT + UNetFormer DSM)",
        "ckpt": "/root/autodl-tmp/runs/plan3_p3r_dsm_20260503_114059/best_model.pt",
        "source": "RS-SAM3-p3r",
        "class": "AdapterSAM3UNetFormerDSM",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
    },
    {
        "name": "VPT+DSM UNetFormer (256² window training)",
        "ckpt": "/root/autodl-tmp/runs/plan3_256win_dsm_20260505_171252/best_model.pt",
        "source": "RS-SAM3-p3r",
        "class": "AdapterSAM3UNetFormerDSM",
        "module": "adapter_unet",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "dropout": 0.1},
        "use_dsm": True,
    },

    # ── Plan3 Route B (RS-SAM-p3b) ────────────────────────
    {
        "name": "LoRA RGB (256² window training)",
        "ckpt": "/root/autodl-tmp/runs/plan3_256win_rgb_20260506_005800/best_model.pt",
        "source": "RS-SAM-p3b",
        "class": "LoRASAM3UNetFormer",
        "module": "lora_sam3",
        "kwargs": {"lora_rank": 8, "lora_alpha": 16, "num_classes": 5, "dropout": 0.1},
        "use_dsm": False,
    },
    {
        "name": "LoRA+DSM (256² window training)",
        "ckpt": "/root/autodl-tmp/runs/plan3_256win_dsm_20260506_102640/best_model.pt",
        "source": "RS-SAM-p3b",
        "class": "LoRASAM3UNetFormer",
        "module": "lora_sam3",
        "kwargs": {"lora_rank": 8, "lora_alpha": 16, "num_classes": 5, "dropout": 0.1, "use_dsm": True},
        "use_dsm": True,
    },

    # ── Plan3 MFNet Decoder (RS-SAM-p3b) ──────────────────
    {
        "name": "VPT + MFNet Decoder (frozen best, 73.10%)",
        "ckpt": "/root/autodl-tmp/runs/plan3_mfnetdec_dsm_20260506_220419/best_model.pt",
        "source": "RS-SAM-p3b",
        "class": "VPT_MFNetDecoder",
        "module": "train_mfnet_decoder",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
    },

    # ── Plan4 Full Training (RS-SAM3-p4) ──────────────────
    {
        "name": "Full SGD (Plan4, epoch 6)",
        "ckpt": "/root/autodl-tmp/runs/plan4_full_vaihingen_20260507_175600/best_model.pt",
        "source": "RS-SAM3-p4",
        "class": "SAM3FullTrain",
        "module": "train_full",
        "kwargs": {"num_classes": 5, "use_dsm": True, "dropout": 0.1},
        "use_dsm": True,
    },
    {
        "name": "D1 unfreeze 8 layers",
        "ckpt": "/root/autodl-tmp/runs/plan4_d8_dsm_20260507_133315/best_model.pt",
        "source": "RS-SAM3-p4",
        "class": "VPT_MFNetDecoder_Unfreeze",
        "module": "train_unfreeze",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True,
                   "dropout": 0.1, "unfreeze_layers": 8},
        "use_dsm": True,
    },
    {
        "name": "D2 unfreeze 16 layers",
        "ckpt": "/root/autodl-tmp/runs/plan4_d16_dsm_20260507_133324/best_model.pt",
        "source": "RS-SAM3-p4",
        "class": "VPT_MFNetDecoder_Unfreeze",
        "module": "train_unfreeze",
        "kwargs": {"adapter_bottleneck": 32, "num_classes": 5, "use_dsm": True,
                   "dropout": 0.1, "unfreeze_layers": 16},
        "use_dsm": True,
    },
]
