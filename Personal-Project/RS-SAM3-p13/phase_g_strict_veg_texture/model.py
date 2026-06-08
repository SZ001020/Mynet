"""P13-G strict vegetation-only texture refinement model.

The P11-B model is frozen. The new head predicts only grass/tree residual logits.
Those residuals are applied strictly inside an oracle or predicted vegetation mask;
outside the mask the final logits are bit-identical to P11-B logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ndsm_roughness import NDSMRoughnessStem
from texture_multiscale import MultiScaleTextureStem


class VegetationBinaryHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_ch, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Plan13GStrictVegTexture(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        resolution: int = 1008,
        use_rgb_texture: bool = True,
        use_ndsm_roughness: bool = False,
        residual_scale: float = 1.0,
        assert_invariant: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.resolution = resolution
        self.use_rgb_texture = use_rgb_texture
        self.use_ndsm_roughness = use_ndsm_roughness
        self.residual_scale = residual_scale
        self.assert_invariant = assert_invariant

        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        feat_ch = 5
        if use_rgb_texture:
            self.texture_stem = MultiScaleTextureStem(in_ch=3, branch_ch=48, out_ch=128)
            feat_ch += 128
        else:
            self.texture_stem = None
        if use_ndsm_roughness:
            self.ndsm_stem = NDSMRoughnessStem(out_ch=64)
            feat_ch += 64
        else:
            self.ndsm_stem = None
        self.veg_head = VegetationBinaryHead(feat_ch, mid_ch=128)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Plan13-G trainable params: {trainable:,}")

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        modules = {"veg_head": self.veg_head.state_dict()}
        if self.texture_stem is not None:
            modules["texture_stem"] = self.texture_stem.state_dict()
        if self.ndsm_stem is not None:
            modules["ndsm_stem"] = self.ndsm_stem.state_dict()
        return modules

    def load_trainable_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.veg_head.load_state_dict(state["veg_head"], strict=True)
        if self.texture_stem is not None and "texture_stem" in state:
            self.texture_stem.load_state_dict(state["texture_stem"], strict=True)
        if self.ndsm_stem is not None and "ndsm_stem" in state:
            self.ndsm_stem.load_state_dict(state["ndsm_stem"], strict=True)

    def _prepare_inputs(self, images: torch.Tensor, dsm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = images
        if rgb.shape[-2:] != (self.resolution, self.resolution):
            rgb = F.interpolate(rgb, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return rgb, dsm

    @staticmethod
    def _oracle_mask(labels: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        labels_small = F.interpolate(labels.unsqueeze(1).float(), out_size, mode="nearest").squeeze(1).long()
        return (labels_small == 2) | (labels_small == 3)

    @staticmethod
    def _pred_mask(base_logits: torch.Tensor) -> torch.Tensor:
        pred = base_logits.argmax(dim=1)
        return (pred == 2) | (pred == 3)

    def forward(
        self,
        images: torch.Tensor,
        dsm: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask_mode: str = "oracle",
    ) -> dict[str, torch.Tensor]:
        if mask_mode not in {"oracle", "pred"}:
            raise ValueError(f"mask_mode must be 'oracle' or 'pred', got {mask_mode}")
        if mask_mode == "oracle" and labels is None:
            raise ValueError("labels are required for oracle vegetation mask")

        with torch.no_grad():
            base_logits = self.base_model(images, dsm).detach()

        out_size = base_logits.shape[-2:]
        rgb_resized, dsm_resized = self._prepare_inputs(images, dsm)
        feats = [base_logits.float()]
        if self.texture_stem is not None:
            feats.append(self.texture_stem(rgb_resized, out_size))
        if self.ndsm_stem is not None:
            feats.append(self.ndsm_stem(dsm_resized, out_size))
        delta = self.veg_head(torch.cat(feats, dim=1)) * self.residual_scale
        refine_grass_tree = base_logits[:, 2:4].float() + delta.float()

        veg_mask = self._oracle_mask(labels, out_size) if mask_mode == "oracle" else self._pred_mask(base_logits)
        veg_mask_2ch = veg_mask.unsqueeze(1).expand_as(refine_grass_tree)

        final_logits = base_logits.float().clone()
        final_logits[:, 2:4] = torch.where(veg_mask_2ch, refine_grass_tree, base_logits[:, 2:4].float())

        if self.assert_invariant:
            nonveg_mask = (~veg_mask).unsqueeze(1).expand_as(final_logits)
            if not torch.equal(final_logits[nonveg_mask], base_logits.float()[nonveg_mask]):
                raise RuntimeError("P13-G invariant failed: non-vegetation logits changed")

        return {
            "final_logits": final_logits,
            "base_logits": base_logits.float(),
            "delta": delta.float(),
            "veg_mask": veg_mask,
            "refine_grass_tree": refine_grass_tree,
        }
