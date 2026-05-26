import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from thop import profile


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
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


class SEConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, stride, padding):
        super(SEConvBlock, self).__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel, stride, padding)
        self.se = SElayer(dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x) + x
        return x


class GaussianConv2d(nn.Module):
    def __init__(self, channels, kernel_size, sigma=1.0):
        super(GaussianConv2d, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.weights = self.get_gaussian_weights()

    def get_gaussian_weights(self):
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)

        return kernel

    def forward(self, x):
        self.weights = self.weights.to(x.device)
        return nn.functional.conv2d(x, self.weights, stride=1, padding=self.padding, groups=self.channels)


class RetinexSSR(nn.Module):
    def __init__(self, dim):
        super(RetinexSSR, self).__init__()
        self.dim = dim
        # layers
        self.conv1 = SEConvBlock(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv4 = SEConvBlock(dim, dim, 3, 1, 1)
        self.conv5 = SEConvBlock(dim, dim, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.gaussian = GaussianConv2d(dim, 3, 1.0)

    def forward(self, x):
        x = self.conv1(x)
        I = self.gaussian(x)
        I = self.conv2(I)
        I = self.sigmoid(I)
        R = self.conv3(x) / I
        R = self.tanh(R)
        R = self.conv4(R) + x
        I = self.conv5(I) + x
        return I, R


class RetinexFusion(nn.Module):
    def __init__(self, dim):
        super(RetinexFusion, self).__init__()
        # parameters
        self.dim = dim
        # blocks
        self.separate1 = RetinexSSR(dim)
        self.separate2 = RetinexSSR(dim)
        self.fusion = SEConvBlock(2 * dim, dim, 1, 1, 0)

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
    x = torch.randn(4, 768, 8, 8).cuda()
    _, c, _, _ = x.shape
    net = RetinexFusion(c).cuda()

    flops, params = profile(net, inputs=(x, x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    result = net(x, x)
    print(result.shape)
