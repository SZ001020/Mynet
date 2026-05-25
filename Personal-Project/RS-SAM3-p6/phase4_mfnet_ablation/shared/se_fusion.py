"""SEFusion: Squeeze-and-Excitation cross-modal fusion (exact MFNet port).

From MFNet paper (IEEE TGRS 2025): SE channel attention applied independently
to RGB and DSM features, then summed element-wise.
"""

import torch.nn as nn


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            activation,
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(nn.functional.adaptive_avg_pool2d(x, 1))


class SEFusion(nn.Module):
    """MFNet-style 4-scale late fusion: SE(RGB) + SE(DSM)."""

    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.se_rgb = SqueezeAndExcitation(channels_in, activation=activation)
        self.se_dsm = SqueezeAndExcitation(channels_in, activation=activation)

    def forward(self, rgb, dsm):
        return self.se_rgb(rgb) + self.se_dsm(dsm)
