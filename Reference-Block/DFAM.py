# 双特征聚合模块（Dual Feature Aggregation Modules, DFAM）
# 核心设计：基于“全局-局部双信息引导+多尺度空洞卷积分支”的架构，
# 通过全局信息（global_info）与局部信息（local_info）双路引导原始特征优化，
# 结合4个差异化分支（1×1卷积+多尺度分解空洞卷积）捕捉不同范围依赖，
# 最终通过特征拼接融合与残差连接，实现全局结构与局部细节的协同增强，提升特征表达的完整性


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Global Contextual module
# DFAM
class DFAM(nn.Module):
    """
    双特征聚合模块（Dual Feature Aggregation Modules, DFAM）
    功能：全局-局部双信息引导+多尺度依赖捕捉，强化特征的全局-局部协同表达
    核心设计：
        - 双信息引导：融合全局信息（全局结构）与局部信息（局部细节），优化原始特征
        - 多尺度分支：4个分支覆盖不同感受野，分别捕捉点级、中短程、中长程、长程依赖
        - 分解空洞卷积：将大核卷积拆分为“1×k + k×1”分组卷积，降低计算成本
        - 残差融合：分支特征拼接后经1×1卷积整合，与残差路径融合，避免信息丢失
    Args:
        in_channel: 输入特征通道数
        out_channel: 输出特征通道数
    """
    def __init__(self, in_channel, out_channel):
        super(DFAM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), groups=out_channel, padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), groups=out_channel, padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, groups=out_channel, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), groups=out_channel, padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), groups=out_channel, padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, groups=out_channel, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), groups=out_channel, padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), groups=out_channel, padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, groups=out_channel, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.conv_trans = BasicConv2d(3 * in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, global_info, local_info):
        B, C, H, W = x.shape
        global_info = F.interpolate(global_info, size=(H, W), mode='bilinear')
        local_info = F.interpolate(local_info, size=(H, W), mode='bilinear')

        x = self.conv_trans(torch.cat([x, x * global_info, x * local_info], dim=1))

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    x = torch.randn(1, 64, 32, 32).to(device)
    t = torch.randn(1, 64, 16, 16).to(device)
    s = torch.randn(1, 64, 16, 16).to(device)
    model = DFAM(64, 64).to(device)

    y = model(x, t, s)



    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)