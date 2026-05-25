"""
LoRA (Low-Rank Adaptation) for SAM3 fine-tuning.

Standard LoRA: adds trainable low-rank decomposition to frozen nn.Linear layers.
  W'x = Wx + (alpha/r) * BAx    where A: d×r, B: r×d, r << d

Supports:
- nn.Linear injection (q_proj, k_proj, v_proj, out_proj, fc1, fc2)
- Selective component targeting (vision_encoder, text_encoder, detr_encoder, etc.)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, List, Optional, Set
import re


class LoRALinear(nn.Module):
    """LoRA-injected nn.Linear wrapper.

    forward: y = Wx + (alpha/rank) * B(A(x))
    where A, B are trainable, W is frozen.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA parameters
        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        if self.training or self.lora_A.requires_grad:
            lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            result = result + lora_out
        return result


def inject_lora_into_module(module: nn.Module, rank: int = 8, alpha: float = 16.0,
                            target_patterns: List[str] = None,
                            exclude_patterns: List[str] = None) -> int:
    """Recursively find nn.Linear layers and wrap them with LoRALinear.

    Args:
        module: root module to search
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_patterns: only inject if the parameter name matches at least one pattern.
                         If None, inject all nn.Linear.
        exclude_patterns: skip if name matches any pattern.

    Returns:
        Number of layers injected.
    """
    if target_patterns is None:
        target_patterns = ['.*']  # match all
    if exclude_patterns is None:
        exclude_patterns = []

    injected = 0
    # Collect all nn.Linear modules
    replacements = {}
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            continue  # already injected
        if isinstance(child, nn.Linear):
            full_name = name
            if any(re.search(p, full_name) for p in exclude_patterns):
                continue
            if any(re.search(p, full_name) for p in target_patterns):
                replacements[name] = child

    for name, original in replacements.items():
        # Navigate to parent and replace
        parts = name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(original, rank=rank, alpha=alpha))
        injected += 1

    return injected


def get_lora_params(module: nn.Module) -> List[nn.Parameter]:
    """Collect all LoRA parameters from a module."""
    params = []
    for name, p in module.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            params.append(p)
    return params


def count_lora_params(module: nn.Module) -> int:
    """Count total LoRA trainable parameters."""
    return sum(p.numel() for p in get_lora_params(module))
