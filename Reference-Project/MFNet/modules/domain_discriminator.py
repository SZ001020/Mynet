import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class PixelDomainDiscriminator(nn.Module):
    """Simple per-pixel domain discriminator for high-level features."""

    def __init__(self, in_channels=256, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, feat):
        return self.net(feat)


class DomainAdversarialHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, grl_lambda=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.discriminator = PixelDomainDiscriminator(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, feat, reverse=False):
        if reverse:
            feat = self.grl(feat)
        return self.discriminator(feat)

    @staticmethod
    def domain_loss(logits, domain_label):
        target = torch.full_like(logits, float(domain_label), device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target)
