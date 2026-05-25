"""
MFNet's complete DFM (Deep Fusion Module) + Decoder for SAM3 pipeline.

Direct port from MFNet paper (IEEE TGRS 2025), adapted for SAM3's ViT output.
- DFM: 4-scale pyramid (1/4, 1/8, 1/16, 1/32) + SEFusion
- Decoder: 3 GLA blocks + FeatureRefinementHead with PA+CA dual attention
- decode_channels=64 (lightweight, matching MFNet's original design)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_


# ═══ MFNet Helpers (exact port) ═══════════════════════════════════

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=pad),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True))  # MFNet uses ReLU6


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=pad),
            norm_layer(out_channels))


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=pad))


class SeparableConvBN(nn.Sequential):
    """MFNet style: BN after depthwise, then pointwise."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=pad, dilation=dilation, groups=in_channels, bias=bias),
            norm_layer(in_channels),  # BN after depthwise on in_channels
            nn.Conv2d(in_channels, out_channels, 1, bias=bias))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x); return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * r.floor_()


# ═══ SEFusion ═══════════════════════════════════════════════════

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class SEFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se_rgb = SqueezeAndExcitation(channels)
        self.se_dsm = SqueezeAndExcitation(channels)

    def forward(self, rgb, dsm):
        return self.se_rgb(rgb) + self.se_dsm(dsm)


# ═══ GLA Attention (MFNet exact port) ══════════════════════════

class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=16, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d((window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d((1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.rel_pos_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing='ij'))
        coords_f = torch.flatten(coords, 1)
        rel = coords_f[:, :, None] - coords_f[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_idx", rel.sum(-1))
        trunc_normal_(self.rel_pos_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.shape
        pw = (ps - W % ps) % ps
        ph = (ps - H % ps) % ps
        if pw > 0 or ph > 0:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        x_p = self.pad(x, self.ws)
        _, _, Hp, Wp = x_p.shape
        qkv = self.qkv(x_p)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
                            h=self.num_heads, d=C // self.num_heads,
                            hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rpb = self.rel_pos_table[self.rel_idx.view(-1)].view(self.ws * self.ws, self.ws * self.ws, -1)
        attn += rpb.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(dim=-1) @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
                         h=self.num_heads, d=C // self.num_heads,
                         hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]

        out = (self.attn_x(F.pad(attn, (0, 0, 0, 1), mode='replicate')) +
               self.attn_y(F.pad(attn, (0, 1, 0, 0), mode='replicate')))
        out = out + local
        out = F.pad(out, (0, 1, 0, 1), mode='replicate')
        out = self.proj(out)[:, :, :H, :W]
        return out


class GLABlock(nn.Module):
    """Pre-norm block: x = x + GLA(norm1(x)); x = x + MLP(norm2(x))"""
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., drop=0., window_size=8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, window_size=window_size)
        self.drop_path = DropPath(drop) if drop > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ═══ Multi-scale Fusion ═══════════════════════════════════════

class WF(nn.Module):
    """Weighted fusion with post-conv refinement (MFNet exact)."""
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        w = torch.relu(self.weights)
        w = w / (torch.sum(w) + self.eps)
        x = w[0] * self.pre_conv(res) + w[1] * x
        return self.post_conv(x)


class FeatureRefinementHead(nn.Module):
    """Top-level refinement with PA (position attention) + CA (channel attention)."""
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        w = torch.relu(self.weights)
        w = w / (torch.sum(w) + self.eps)
        x = w[0] * self.pre_conv(res) + w[1] * x
        x = self.post_conv(x)

        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        return self.act(x)


# ═══ MFNet Decoder (4-scale, exact port) ═════════════════════

class MFNetDecoder(nn.Module):
    """MFNet's Decoder class, exactly as in the paper.

    Input: 4-scale features [res1@1/4, res2@1/8, res3@1/16, res4@1/32]
    Each res_i has encoder_channels[i] channels (all 256 in MFNet).
    decode_channels = 64 (matches MFNet's lightness).
    """

    def __init__(self, num_classes=5, decode_channels=64, dropout=0.1, window_size=8,
                 encoder_channels=(256, 256, 256, 256)):
        super().__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b3 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.b2 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.seg_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(dropout, inplace=False),
            Conv(decode_channels, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, feats, output_size=None):
        """feats: [res1@1/4, res2@1/8, res3@1/16, res4@1/32]"""
        res1, res2, res3, res4 = feats

        x = self.b4(self.pre_conv(res4))             # 1/32
        x = self.p3(x, res3); x = self.b3(x)          # 1/16
        x = self.p2(x, res2); x = self.b2(x)          # 1/8
        x = self.p1(x, res1)                          # 1/4
        x = self.seg_head(x)                          # 1/4

        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x


# ═══ 4-Scale Pyramid Builder (SAM3 → MFNet scales) ═══════════

class Pyramid4Scale(nn.Module):
    """Build MFNet's 4-scale pyramid from SAM3's ViT output.

    SAM3 ViT outputs a single feature map (256ch, ~63² for 1008 input).
    This expands it to 4 scales: 1/4, 1/8, 1/16, 1/32 via ConvTranspose / pooling.
    """

    def __init__(self, in_channels=256):
        super().__init__()
        # 1/4 (upsample 4x from ~1/16 = 16x16 to 64x64)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
        )
        # 1/8 (upsample 2x)
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
        )
        # 1/16 (identity)
        self.fpn3 = nn.Identity()
        # 1/32 (downsample 2x)
        self.fpn4 = nn.MaxPool2d(2, 2)

    def forward(self, feat_1_16):
        """feat_1_16: (B, 256, H/16, W/16) from SAM3 ViT"""
        s1 = self.fpn1(feat_1_16)  # 1/4
        s2 = self.fpn2(feat_1_16)  # 1/8
        s3 = self.fpn3(feat_1_16)  # 1/16
        s4 = self.fpn4(feat_1_16)  # 1/32
        return [s1, s2, s3, s4]
