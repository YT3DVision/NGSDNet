import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DirectionalConvUnit(nn.Module):
    # modified from the basic version of GeleNet (https://arxiv.org/abs/2309.08206)
    def __init__(self, channel, kernel, dilation):
        super(DirectionalConvUnit, self).__init__()
        # parameters
        self.in_dim = channel
        self.out_dim = channel
        self.kernal_size = kernel  # 3,5,7,9
        self.dilation = dilation  # 1,2,3,4
        self.padding = dilation * dilation  # 1,4,9,16

        # layers
        self.h_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (1, self.kernal_size),
                      padding=(0, self.padding), dilation=(1, self.dilation)),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU()
        )

        self.w_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (self.kernal_size, 1),
                      padding=(self.padding, 0), dilation=(self.dilation, 1)),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU()
        )
        # leading diagonal
        self.dia19_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (self.kernal_size, 1),
                      padding=(self.padding, 0), dilation=(self.dilation, 1)),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU()
        )
        # reverse diagonal
        self.dia37_conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, (1, self.kernal_size),
                      padding=(0, self.padding), dilation=(1, self.dilation)),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU()
        )

        # fusion layer
        self.fusion = SKFF(self.out_dim, 4, 4)

    def forward(self, x):
        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        # x = torch.cat((x1, x2, x3, x4), 1)
        x = [x1, x2, x3, x4]
        x = self.fusion(x)

        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)


class SKFF(nn.Module):
    # reference to MIRNet (https://github.com/swz30/MIRNet)
    def __init__(self, in_channels, height=4, reduction=4):
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels // reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, stride=1, padding=0), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, padding=0))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        in_channels = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, in_channels, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, in_channels, 1, 1)

        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class SElayer(nn.Module):
    # reference to Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
    def __init__(self, channel, reduction=8):
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


class DSFEM(nn.Module):
    def __init__(self, channel):
        super(DSFEM, self).__init__()
        self.dim = channel

        # channel reduction
        self.cr = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, 1, 1, 1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU()
        )

        # dconv
        self.DirectionalConv3 = DirectionalConvUnit(channel//4, 3, 1)
        self.DirectionalConv5 = DirectionalConvUnit(channel//4, 5, 2)
        self.DirectionalConv7 = DirectionalConvUnit(channel//4, 7, 3)
        self.DirectionalConv9 = DirectionalConvUnit(channel//4, 9, 4)

        # fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )
        self.se_layer = SElayer(self.dim, 8)

    def forward(self, x):
        cr = self.cr(x)
        x1 = self.DirectionalConv3(cr)
        x2 = self.DirectionalConv5(cr)
        x3 = self.DirectionalConv7(cr)
        x4 = self.DirectionalConv9(cr)

        # fusion stage
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.fusion_conv(x)
        x = x + self.se_layer(x)

        return x
