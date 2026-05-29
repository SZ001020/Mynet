# 网络整体架构以及模块Cross Attention and Invertible Block (CAI)：CAI 对红外 / 可见光单模态特征做边缘感知的基础特征提取，通过双向交叉注意力实现双模态特征的互补增强，再通过可逆块实现特征的无损变换与深度融合。
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# -------------------------- 基础卷积模块 --------------------------
class ConvLeakyRelu2d(nn.Module):
    """带LeakyReLU的基础卷积块，用于密集特征提取"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    """
    Sobel边缘提取算子 对应结构图(d) BFE模块的边缘特征提取
    核心功能：显式提取图像的水平+垂直边缘信息，保留结构细节
    """
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # 水平方向边缘卷积
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        # 垂直方向边缘卷积
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    """1×1点卷积，用于通道变换与特征融合"""
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    """密集连接块，用于多尺度特征提取，提升特征复用率"""
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x

# -------------------------- 对应结构图(d) BFE基础特征提取模块 --------------------------
class RGBD(nn.Module):
    """
    RGBD边缘感知特征提取块 对应结构图(d) Base Feature Extraction Block
    核心功能：密集块提取语义特征 + Sobel算子提取边缘特征，融合得到结构-语义联合特征
    """
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        # 语义特征分支
        x1=self.dense(x)
        x1=self.convdown(x1)
        # 边缘特征分支
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        # 双分支融合
        return F.leaky_relu(x1+x2,negative_slope=0.1)

# -------------------------- 可逆块基础单元 对应结构图(c)可逆变换部分 --------------------------
class InvertedResidualBlock(nn.Module):
    """
    倒残差块 可逆变换的基础单元
    采用pw+dw+pw的轻量化结构，保证特征变换的可逆性与低计算量
    """
    def __init__(self, inp=32, oup=32, expand_ratio=2):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

# -------------------------- 对应结构图(c) 可逆特征变换节点 --------------------------
class DetailNode(nn.Module):
    """
    可逆特征变换节点 对应结构图(c) Invertible Block核心部分
    核心功能：通过仿射变换实现特征的无损可逆增强，保证两阶段训练的特征一致性
    """
    def __init__(self):
        super(DetailNode, self).__init__()
        # 可逆仿射变换的三个可学习参数矩阵
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        """将特征拆分为两个等通道分支，适配可逆变换"""
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        # 特征混洗与拆分
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        # 可逆仿射变换：z2 = z2 + φ(z1)
        z2 = z2 + self.theta_phi(z1)
        # 可逆仿射变换：z1 = z1 * exp(ρ(z2)) + η(z2)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

# -------------------------- 对应结构图(c) 交叉注意力核心模块 --------------------------
class Cross_Trans_Attention(nn.Module):
    """
    跨模态交叉注意力模块 对应结构图(c) Cross Attention核心部分
    核心功能：以guide特征为引导，对op特征做自适应增强，实现双模态特征的互补交互
    """
    def __init__(self,num_heads,dim):
        super(Cross_Trans_Attention, self).__init__()
        self.num_heads = num_heads
        bias=True
        # 可学习温度系数，控制注意力分布平滑度
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # KV生成：来自被引导的特征
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # Q生成：来自引导特征
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 深度卷积增强空间特征
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_guide,feat_op):
        """
        :param feat_guide: 引导特征（如红外结构特征/可见光纹理特征）
        :param feat_op: 被引导的待增强特征
        :return: 引导增强后的特征
        """
        b,c,h,w = feat_guide.shape
        # Q/K/V生成与空间增强
        q = self.q_dwconv(self.q(feat_guide))
        kv = self.kv_dwconv(self.kv(feat_op))
        k,v = kv.chunk(2, dim=1)
        # 维度重整为多头注意力格式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # L2归一化提升注意力稳定性
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 通道级交叉注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 注意力加权输出
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# -------------------------- CAI完整模块 对应结构图(a)(c) Cross Attention and Invertible Block --------------------------
class DetailFeatureExtraction_Encoder(nn.Module):
    """
    CAI: Cross Attention and Invertible Block 完整实现
    完整对应结构图(c)全流程，是结构图(a)中双模态编码器的核心模块
    核心功能：边缘感知特征提取 → 双向交叉模态注意力 → 可逆无损特征增强
    """
    def __init__(self, dim=32, num_layers=1):
        super(DetailFeatureExtraction_Encoder, self).__init__()
        # 边缘感知基础特征提取（BFE）
        self.GRDB = RGBD(in_channels=dim, out_channels=dim)
        # 双向交叉注意力模块
        self.TCA = Cross_Trans_Attention(num_heads=8, dim=dim // 2)
        # 可逆特征变换块
        self.layer = DetailNode()

    def forward(self, x):
        # 步骤1：边缘感知基础特征提取
        x = self.GRDB(x)
        # 步骤2：特征拆分为两个分支，对应双模态特征
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        # 步骤3：双向交叉注意力：z2引导z1、z1引导z2，实现双模态互补增强
        feat1 = self.TCA(feat_guide=z2, feat_op=z1)
        feat2 = self.TCA(feat_guide=z1, feat_op=z2)
        # 步骤4：可逆块无损特征增强
        feat1, feat2 = self.layer(feat1, feat2)
        # 步骤5：特征拼接输出
        return torch.cat((feat1, feat2), dim=1)


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = DetailFeatureExtraction_Encoder(64).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)