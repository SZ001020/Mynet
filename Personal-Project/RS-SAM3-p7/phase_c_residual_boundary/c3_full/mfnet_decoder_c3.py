"""MFNetDecoder variant for Plan7-C3 that exposes intermediate b3 features.

Same architecture as Plan7-A's MFNetDecoder, but stores the b3 GLABlock output
as an attribute for use by the SAM-HQ GlobalLocalFusion.
"""

import sys, os

BASE = "/root/Mynet"
PHASE_A = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
sys.path.insert(0, PHASE_A)

import torch.nn.functional as F
from mfnet_decoder import MFNetDecoder as _BaseDecoder


class MFNetDecoderC3(_BaseDecoder):
    """MFNetDecoder that exposes b3 intermediate output.

    After each forward call, self.b3_output holds the feature map
    at 1/16 resolution (after b3 GLABlock, before p2 upsampling).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b3_output = None

    def forward(self, feats, output_size=None):
        res1, res2, res3, res4 = feats

        x = self.b4(self.pre_conv(res4))     # 1/32
        x = self.p3(x, res3); x = self.b3(x)  # 1/16
        self.b3_output = x                     # store for C3

        x = self.p2(x, res2); x = self.b2(x)   # 1/8
        x = self.p1(x, res1)                   # 1/4
        x = self.seg_head(x)                   # 1/4

        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x
