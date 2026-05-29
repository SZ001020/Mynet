import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class downsample_vit(nn.Module):
    """
    带局部增强位置编码(LEPE)的窗口自注意力模块
    对应结构图(c)的Attention单元，为每个尺度分支提供窗口自注意力计算
    Args:
        dim: 单分支输入通道数
        window_size: 窗口划分尺寸，默认8
        attn_drop: 注意力dropout率
        proj_drop: 输出投影dropout率
        down_scale: 对应分支的下采样倍率，用于适配不同尺度特征
    """

    def __init__(self, dim, window_size=8, attn_drop=0., proj_drop=0., down_scale=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.scale = dim ** -0.5# 注意力缩放系数

        # QKV线性投影层，对应结构图(c)中的Linear层
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 局部增强位置编码(LEPE)，对应结构图(c)中的PE模块，用深度卷积捕捉局部空间信息
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x, window_size):
        """将特征图划分为不重叠的窗口，适配窗口注意力计算"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    def window_reverse(self, windows, window_size, h, w):
        """将窗口特征还原为完整特征图"""
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def get_lepe(self, x, func):
        """生成局部增强位置编码LEPE，为value特征添加空间位置信息"""
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        # 窗口划分，适配深度卷积
        H_sp, W_sp = self.window_size, self.window_size
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)

        # 深度卷积生成位置编码
        lepe = func(x)
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x):
        """窗口自注意力前向传播，对应结构图(c)的完整流程"""
        B, C, H, W = x.shape
        # 1. 维度转换：CHW → HWC，适配窗口划分
        x = x.permute(0, 2, 3, 1)
        # 2. 窗口划分
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(-1, self.window_size * self.window_size, C)
        # 3. 生成QKV三分支
        qkv = self.qkv(x_window)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 4. 为value添加局部增强位置编码
        v, lepe = self.get_lepe(v, self.get_v)
        # 5. 窗口自注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 6. 注意力输出 + 位置编码残差增强
        x = (attn @ v) + lepe
        # 7. 输出投影与dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        # 8. 窗口还原为完整特征图
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)
        # 维度还原为CHW格式，适配后续操作
        return x.permute(0, 3, 1, 2)


class LMLT(nn.Module):
    """
    Low-to-high Multi-Level Transformer 核心模块
    对应结构图(b)的完整流程：通道拆分→多尺度下采样→低到高反向注意力计算→跨层级融合→多尺度聚合→门控输出
    Args:
        dim: 输入特征总通道数
        attn_drop: 注意力dropout率
        proj_drop: 输出投影dropout率
        n_levels: 多尺度层级数，默认4，对应结构图中的DS/1、DS/2、DS/4、DS/8四个分支
    """

    def __init__(self, dim, attn_drop=0., proj_drop=0., n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        self.chunk_dim = dim // n_levels  # 每个层级分配的通道数，对应结构图中的Split拆分

        # 每个层级对应一个窗口自注意力分支，对应结构图中的多个Attention模块
        self.mfr = nn.ModuleList([
            downsample_vit(
                dim // 4, window_size=8, attn_drop=attn_drop, proj_drop=proj_drop, down_scale=2 ** i
            ) for i in range(self.n_levels)
        ])

        # 多尺度特征聚合卷积，对应结构图中的Concat+Conv聚合
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # 激活函数与门控
        self.act = nn.GELU()

    def forward(self, x):
        """LMLT完整前向流程，严格对应结构图(b)的Low-to-high逻辑"""
        h, w = x.size()[-2:]
        # 步骤1：通道维度拆分，对应结构图中的Split，将输入拆分为n_levels个独立分支
        xc = x.chunk(self.n_levels, dim=1)
        downsampled_feat = []

        # 步骤2：多尺度下采样，对应结构图中的DS/1、DS/2、DS/4、DS/8，生成多分辨率特征
        for i in range(self.n_levels):
            if i > 0:
                # 层级越高，下采样倍率越大，分辨率越低
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                downsampled_feat.append(s)
            else:
                # 第0层保持原始分辨率，无下采样
                downsampled_feat.append(xc[i])

        out = []
        # 步骤3：Low-to-high 自底向上处理（核心创新），从最低分辨率开始，反向向上融合
        for i in reversed(range(self.n_levels)):
            # 对当前层级特征做窗口自注意力计算
            s = self.mfr[i](downsampled_feat[i])
            # 2倍上采样，与上一层级特征做残差融合，对应结构图中的US*2
            s_upsample = F.interpolate(s, size=(s.shape[2] * 2, s.shape[3] * 2), mode='nearest')
            if i > 0:
                # 低分辨率语义信息上采样后，引导高分辨率细节特征
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample
            # 将当前层级特征上采样回原始输入尺寸，用于最终聚合
            s_original_shape = F.interpolate(s, size=(h, w), mode='nearest')
            out.append(s_original_shape)

        # 步骤4：多尺度特征聚合，对应结构图中的Concat+Conv
        out = self.aggr(torch.cat(out, dim=1))
        # 步骤5：门控残差输出，聚合后的特征与原特征相乘，实现全局特征增强
        out = self.act(out) * x
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
    x = torch.randn(1, 64, 64, 64).to(device)
    model = LMLT(64).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)