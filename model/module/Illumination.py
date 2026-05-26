import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from thop import profile

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

def IllumiDown(dim_in, dim_out):
    return BasicConv2d(dim_in, dim_out, 4, 2, 1)

class Illumination(nn.Module):
    def __init__(self, dim_middle, dim_in=3, dim_out=3, img_size=256):
        super(Illumination, self).__init__()
        # parameters
        self.dim_in = dim_in
        self.dim_middle = dim_middle
        self.dim_out = dim_out
        self.img_size = img_size

        self.conv1 = BasicConv2d(dim_in+1, dim_middle//4, 1, 1, 0)
        self.conv2 = DepthWiseConv2d(dim_middle//4, dim_middle//4, 3, 1, 1)
        self.conv3 = BasicConv2d(dim_middle//4, dim_out, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d((img_size, img_size))

        self.down1 = BasicConv2d(dim_middle // 4, dim_middle // 2, 4, 2, 1)
        self.down2 = BasicConv2d(dim_middle // 2, dim_middle, 4, 2, 1)

    def forward(self, x):
        if self.dim_in == 3:
            mean_c = x.mean(dim=1).unsqueeze(1)
            mean_c = self.avg_pool(mean_c)
        else:
            mean_c = self.avg_pool(x)
        input = torch.cat([x, mean_c], dim=1)
        input = self.conv1(input)
        illu_fea = self.conv2(input)
        illu_map = self.conv3(illu_fea)

        # downsample
        illu_fea = self.down1(illu_fea)
        illu_fea = self.down2(illu_fea)

        return illu_fea, illu_map


# for debug
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).cuda()
    net = Illumination(96, 3,3, 256).cuda()

    # print params and flops
    flops, params = profile(net, inputs=(x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    illu_fea, illU_map = net(x)
    print(illu_fea.shape)
    print(illU_map.shape)
