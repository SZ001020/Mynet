"""
Plan3 Route A: Adapter-ViT — 在 SAM3 ViT backbone 的每个 Block 前注入
trainable prompt perturbation（VPT 变体），保持 ViT 本体 frozen。

Adapter: 1024→32→1024 bottleneck MLP, ~65K params each
32 blocks × 65K ≈ 2M trainable params
"""

import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    """Wrap a frozen ViT Block with a trainable prompt_learn MLP.

    forward:  x → x + prompt_learn(x) → frozen_block → output

    The prompt_learn MLP is a bottleneck (dim→bottleneck→dim) that learns
    a token-wise feature perturbation for remote sensing adaptation.
    """

    def __init__(self, block: nn.Module, dim: int = 1024, bottleneck: int = 32):
        super().__init__()
        self.block = block  # frozen ViT Block

        # Freeze the original block
        for p in self.block.parameters():
            p.requires_grad = False

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prompt_learn tracks gradients normally — since block params have
        # requires_grad=False, backprop flows through block to prompt_learn
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


def inject_adapters(vision_backbone: nn.Module, bottleneck: int = 32) -> nn.Module:
    """Replace all ViT blocks in the SAM3 vision backbone with AdapterBlock wrappers.

    Modifies the model in-place and returns it.
    Only the Adapter's prompt_learn parameters are trainable;
    all ViT blocks and other backbone components remain frozen.

    Args:
        vision_backbone: SAM3's vision_backbone (Sam3DualViTDetNeck)
        bottleneck: adapter bottleneck dimension (default 32 → 1024/32 = 32:1 compression)

    Returns:
        The modified vision_backbone (same object, in-place)
    """
    blocks = vision_backbone.trunk.blocks
    dim = blocks[0].attn.qkv.in_features  # 1024
    replaced = 0

    for i, blk in enumerate(blocks):
        if not isinstance(blk, AdapterBlock):
            blocks[i] = AdapterBlock(blk, dim=dim, bottleneck=bottleneck)
            replaced += 1

    print(f"  Injected {replaced} AdapterBlocks (dim={dim}, bottleneck={bottleneck})")

    # Freeze everything except adapter prompt_learn params
    for name, p in vision_backbone.named_parameters():
        if 'prompt_learn' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in vision_backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vision_backbone.parameters())
    print(f"  Vision backbone: {trainable:,} / {total:,} trainable params ({trainable/total*100:.1f}%)")

    return vision_backbone
