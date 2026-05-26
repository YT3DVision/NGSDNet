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


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=1):
        super(DepthWiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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


class SEConvBlock(nn.Module):
    def __init__(self, dim, kernel, stride, padding):
        super(SEConvBlock, self).__init__()
        self.conv = DepthWiseConv2d(dim, dim, kernel, stride, padding)
        self.se = SElayer(dim)

    def forward(self, x):
        return x + self.se(self.conv(x))


class CrossPool(nn.Module):
    def __init__(self, dim):
        super(CrossPool, self).__init__()
        self.dim = dim
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.fusion = BasicConv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, hf, lf):
        hf_p = self.maxpool(hf)
        lf_p = self.avgpool(lf)
        hf_p = F.interpolate(hf_p, scale_factor=2, mode='bilinear', align_corners=True)
        lf_p = F.interpolate(lf_p, scale_factor=2, mode='bilinear', align_corners=True)
        map = torch.sigmoid(hf_p - lf_p)
        residual_hf = map * hf - (1 - map) * hf_p
        residual_lf = map * lf - (1 - map) * lf_p
        x = torch.cat((residual_hf, residual_lf), dim=1)
        x = self.fusion(x)
        return x


class CrossFusion(nn.Module):
    def __init__(self, dim):
        super(CrossFusion, self).__init__()
        self.dim = dim
        self.se1 = SEConvBlock(dim, 3, 1, 1)
        self.se2 = SEConvBlock(dim, 3, 1, 1)
        self.se3 = SEConvBlock(2 * dim, 3, 1, 1)

        self.cross1 = CrossPool(dim)
        self.cross2 = CrossPool(dim)

    def forward(self, rgb, nir):
        rgb = self.se1(rgb)
        nir = self.se2(nir)
        map = torch.cat((self.cross1(rgb, nir), self.cross2(nir, rgb)), 1)
        map = self.se3(map)
        map = torch.sigmoid(map)
        rgb = rgb * map[:, :self.dim, :, :]
        nir = nir * map[:, self.dim:, :, :]
        return rgb * nir


# for debug
if __name__ == '__main__':
    x = torch.randn(1, 768, 8, 8).cuda()
    b, c, h, w = x.shape
    net = CrossFusion(c).cuda()

    flops, params = profile(net, inputs=(x, x))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    result = net(x, x)
    print(result.shape)