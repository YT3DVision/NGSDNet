import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from thop import profile


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to: Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class BasicConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=1, dilation=1):
        super(DepthWiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=dim_in,
        )
        self.BN1 = nn.BatchNorm2d(dim_in)
        self.act1 = nn.ReLU()
        self.point_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        self.BN2 = nn.BatchNorm2d(dim_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.BN1(x)
        x = self.act1(x)
        x = self.point_conv(x)
        x = self.BN2(x)
        out = self.act2(x)
        return out


class FrequencySeparate(nn.Module):
    def __init__(self, dim):
        super(FrequencySeparate, self).__init__()
        # parameters
        self.dim = dim
        # blocks
        self.se1 = SElayer(dim)
        self.se2 = SElayer(dim)
        self.se3 = SElayer(dim)
        self.se4 = SElayer(dim)

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Stage1:AvgPool
        lf = self.avg_pool(x)
        hf = x - F.interpolate(lf, scale_factor=2, mode='nearest')
        hf = self.se1(hf) + hf
        lf = self.se2(lf) + lf
        lf = F.interpolate(lf, scale_factor=2, mode='nearest')
        x = torch.cat((hf, lf), dim=1)
        x = self.fusion(x)

        # Stage2:MaxPool
        lf = self.max_pool(lf)
        hf = x - F.interpolate(lf, scale_factor=2, mode='nearest')
        hf = self.se3(hf) + hf
        lf = self.se4(lf) + lf
        lf = F.interpolate(lf, scale_factor=2, mode='nearest')

        return lf, hf


class FrequencyFusion(nn.Module):
    def __init__(self, dim):
        super(FrequencyFusion, self).__init__()
        # parameters
        self.dim = dim
        # blocks
        self.separate1 = FrequencySeparate(dim)
        self.separate2 = FrequencySeparate(dim)
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb, nir):
        lf_rgb, hf_rgb = self.separate1(rgb)
        lf_nir, hf_nir = self.separate2(nir)

        map = torch.sigmoid(abs(hf_rgb - hf_nir))
        residual_rgb = map * lf_rgb - (1 - map) * hf_rgb
        residual_nir = map * lf_nir - (1 - map) * hf_nir
        x = torch.cat((residual_rgb, residual_nir), dim=1)
        x = self.fusion(x)

        return x

# for debug
if __name__ == '__main__':
    x = torch.randn(1, 96, 64, 64).cuda()
    net = FrequencyFusion(96).cuda()

    flops, params = profile(net, inputs=(x, x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    result = net(x, x)
    print(result.shape)
