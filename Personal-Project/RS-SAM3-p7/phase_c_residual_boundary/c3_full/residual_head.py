"""SAM-HQ style residual correction head for Plan7-C3.

Lightweight conv head (~37K params) that predicts residual logits
added to the main decoder output (logit-level correction, not mask-level).
"""

import torch.nn as nn


class ResidualCorrectionHead(nn.Module):
    """HQ-Features → residual logits. ~37K params.

    Follows SAM-HQ's minimal correction design:
    - 2 conv layers + dropout
    - Outputs residual logits (NOT masks)
    - Element-wise addition with main logits happens in the model forward
    """

    def __init__(self, in_channels=64, num_classes=5, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(dropout, inplace=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, hq_features):
        return self.head(hq_features)
