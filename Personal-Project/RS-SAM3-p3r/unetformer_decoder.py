"""
UNetFormer-style Decoder with Global-Local Attention (GLA) blocks.
Adapted from MFNet (IEEE TGRS 2025) for 3-scale FPN inputs.

Key components:
- GlobalLocalAttention: window-based self-attention + local conv branch
- Block: Pre-norm Transformer block with GLA
- WF_single: Weighted fusion for multi-scale skip connections
- Decoder: 3-level decoder (72→144→288→576) replacing simple UNet ConvBlocks
"""

import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_


# ── Helpers ──────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=pad),
            norm_layer(out_channels),
            nn.ReLU(inplace=False))

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=pad, dilation=dilation, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            norm_layer(out_channels))

class Mlp(nn.Module):
    def __init__(self, in_f, hidden_f=None, out_f=None, act=nn.ReLU6, drop=0.):
        super().__init__()
        out_f = out_f or in_f; hidden_f = hidden_f or in_f
        self.fc1 = nn.Conv2d(in_f, hidden_f, 1); self.act = act()
        self.fc2 = nn.Conv2d(hidden_f, out_f, 1); self.drop = nn.Dropout2d(drop)

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
    def extra_repr(self): return f'drop_prob={self.drop_prob}'


# ── Global-Local Attention (core of UNetFormer) ─────────────

class GlobalLocalAttention(nn.Module):
    """Window-based self-attention + local conv branch.

    Splits feature map into 8×8 windows, does self-attention within each window,
    then combines with a depthwise local conv branch.
    """

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

        # Cross-shaped attention aggregation
        self.attn_x = nn.AvgPool2d((window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d((1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        # Relative position bias
        self.rel_pos_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)]))
        coords_f = torch.flatten(coords, 1)
        rel = coords_f[:, :, None] - coords_f[:, None, :]  # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_idx = rel.sum(-1)
        self.register_buffer("rel_idx", rel_idx)
        trunc_normal_(self.rel_pos_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.shape
        if W % ps != 0: x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0: x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
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

        out = (self.attn_x(F.pad(attn, (0, 0, 0, 1), mode='reflect')) +
               self.attn_y(F.pad(attn, (0, 1, 0, 0), mode='reflect')))

        out = out + local
        out = F.pad(out, (0, 1, 0, 1), mode='reflect')
        out = self.proj(out)[:, :, :H, :W]
        return out


# ── Transformer Block with GLA ───────────────────────────────

class GLABlock(nn.Module):
    """Pre-norm block: x = x + GLA(norm1(x)); x = x + MLP(norm2(x))"""

    def __init__(self, dim=256, num_heads=8, mlp_ratio=4., drop=0., window_size=8):
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


# ── Multi-scale Fusion Modules ──────────────────────────────

class WFSingle(nn.Module):
    """Weighted fusion: fuse skip connection from higher resolution."""

    def __init__(self, in_ch, decode_ch):
        super().__init__()
        self.pre = Conv(in_ch, decode_ch, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x, skip):
        skip = self.pre(skip)
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w = torch.relu(self.weights)
        return (w[0] * x + w[1] * skip) / (w[0] + w[1] + 1e-8)


class FeatureRefinementHead(nn.Module):
    """Final upsampling: fuse with highest-res feature → 2× upsample."""

    def __init__(self, in_ch, decode_ch):
        super().__init__()
        self.pre = Conv(in_ch, decode_ch, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.refine = nn.Sequential(ConvBNReLU(decode_ch, decode_ch),
                                    ConvBNReLU(decode_ch, decode_ch),
                                    nn.Conv2d(decode_ch, decode_ch, 1))

    def forward(self, x, skip):
        skip = self.pre(skip)
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w = torch.relu(self.weights)
        x = (w[0] * x + w[1] * skip) / (w[0] + w[1] + 1e-8)
        return self.refine(x)


# ── UNetFormer Decoder ──────────────────────────────────────

class UNetFormerDecoder(nn.Module):
    """Deep UNetFormer decoder with stacked GLA blocks per scale.

    FPN inputs: [288²@256, 144²@256, 72²@256]
    Each scale: 2 stacked GLABlocks for deeper feature refinement.
    """

    def __init__(self, num_classes=5, decode_channels=256, dropout=0.1, window_size=8,
                 encoder_channels=(256, 256, 256)):
        super().__init__()
        # encoder_channels: (res2@288², res3@144², res4@72²)
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        # Deepest level (72²)
        self.b4 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # 144² level: WF fuse + GLA
        self.p3 = WFSingle(encoder_channels[-2], decode_channels)
        self.b3 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # 288² level: WF fuse + GLA
        self.p2 = WFSingle(encoder_channels[-3], decode_channels)
        self.b2 = GLABlock(decode_channels, num_heads=8, drop=dropout, window_size=window_size)

        # Final refinement + head
        self.p1 = FeatureRefinementHead(encoder_channels[-3], decode_channels)
        self.head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(dropout, inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv(decode_channels, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        """feats: [f0@288², f1@144², f2@72²]"""
        f0, f1, f2 = feats

        # Bottleneck: 72² → GLA
        x = self.b4(self.pre_conv(f2))

        # 72² → 144², fuse with f1, GLA
        x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.p3(x, f1)
        x = self.b3(x)

        # 144² → 288², fuse with f0, GLA
        x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        x = self.p2(x, f0)
        x = self.b2(x)

        # Refine + fuse with f0 again + upsample
        x = self.p1(x, f0)
        return self.head(x)
