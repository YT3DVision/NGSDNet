import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.backbone.swin_transformer.swin_transformer_v2 import SwinTransformerV2_demo
from thop import profile

from model.module.Attention import CrossAttentionTransformer

def add_conv_stage(
    dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False
):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(),
        )


def upsample(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        # nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ConvTranspose2d(
            ch_coarse, ch_fine, kernel_size, stride, padding, bias=False
        ),
        nn.ReLU(),
    )


def downsample(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            ch_coarse, ch_fine, kernel_size, stride, padding, bias=False
        ),
        nn.ReLU(),
    )


class Conv2DLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        k_size,
        stride,
        padding=None,
        dilation=1,
        norm=1,
        act=1,
        bias=False,
    ):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module(
            "conv2d",
            nn.Conv2d(
                in_channels,
                out_channels,
                k_size,
                stride,
                padding,
                dilation=dilation,
                bias=bias,
            ),
        )
        if norm is not None:
            self.add_module("norm", norm(out_channels))
        if act is not None:
            self.add_module("act", act)


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class RetinexFusion(nn.Module):
    def __init__(self, dim):
        super(RetinexFusion, self).__init__()
        self.conv1 = DepthWiseConv2d(2 * dim, dim, 3, 1, 1)
        self.conv2 = DepthWiseConv2d(dim, dim, 3, 1, 1)
        self.se1 = SElayer(dim)
        self.se2 = SElayer(dim)
        self.attn1 = CrossAttentionTransformer(dim)
        self.attn2 = CrossAttentionTransformer(dim)
        self.fusion = BasicConv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, rgb, nir, rt_fea):
        rgb = self.se1(self.conv1(torch.cat([rgb, rt_fea], dim=1)))
        nir = self.se2(self.conv2(nir))
        map = torch.sigmoid(rgb - nir)
        rgb = self.attn1(rgb, rgb * map)
        nir = self.attn2(nir, nir * map)
        x = torch.cat([rgb, nir], dim=1)
        x = self.fusion(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthWiseConv2d(nn.Module):
    #  Depth-wise Separable Convolution Layer
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, dilation=1):
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
        x = self.act2(x)
        return x


class RetinexBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_middle):
        super(RetinexBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_middle = dim_middle
        self.dim_out = dim_out

        self.conv1 = BasicConv2d(dim_in, dim_middle, 1, 1, 0)
        self.conv2 = DepthWiseConv2d(dim_middle, dim_middle, 5, 1, 2)
        self.conv3 = BasicConv2d(dim_middle, dim_out, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        rt_fea = self.conv2(x)
        rt_map = self.conv3(rt_fea)
        return rt_map, rt_fea


class RetinexNet(nn.Module):
    def __init__(
        self,
        backbone_path=None,
    ):
        super(RetinexNet, self).__init__()
        swin_transformer = SwinTransformerV2_demo(
            img_size=256,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            window_size=16,
        )

        swin_transformer2 = SwinTransformerV2_demo(
            img_size=256,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            window_size=16,
        )

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict["model"]
            print("---start load pretrained model of swin encoder---")
            swin_transformer.load_state_dict(pretrained_dict, strict=False)
            swin_transformer2.load_state_dict(pretrained_dict, strict=False)

        self.pebed = swin_transformer.patch_embed
        self.pos_drop = swin_transformer.pos_drop
        self.rgb_layer0 = swin_transformer.layers[0]
        self.rgb_layer1 = swin_transformer.layers[1]
        self.rgb_layer2 = swin_transformer.layers[2]
        self.rgb_layer3 = swin_transformer.layers[3]

        self.pebed = swin_transformer2.patch_embed
        self.pos_drop = swin_transformer2.pos_drop
        self.nir_layer0 = swin_transformer2.layers[0]
        self.nir_layer1 = swin_transformer2.layers[1]
        self.nir_layer2 = swin_transformer2.layers[2]
        self.nir_layer3 = swin_transformer2.layers[3]

        # backbone的参数
        embed_dim = 96
        img_size = 256

        # retinex
        self.retinex = RetinexBlock(3, 3, embed_dim // 4)
        self.down0 = downsample(embed_dim // 4, embed_dim, 8, 4, 2)
        self.down1 = downsample(embed_dim, embed_dim * 2)
        self.down2 = downsample(embed_dim * 2, embed_dim * 4)
        self.down3 = downsample(embed_dim * 4, embed_dim * 8)

        # mcc
        self.mcc1 = RetinexFusion(embed_dim * 8)
        self.mcc2 = RetinexFusion(embed_dim * 4)
        self.mcc3 = RetinexFusion(embed_dim * 2)
        self.mcc4 = RetinexFusion(embed_dim)

        self.up_32 = upsample(embed_dim * 8, embed_dim * 4)
        self.up_21 = upsample(embed_dim * 4, embed_dim * 2)
        self.up_10 = upsample(embed_dim * 2, embed_dim)
        self.up_final = upsample(embed_dim, embed_dim // 2, kernel_size=8, stride=4, padding=2)

        self.conv3m = add_conv_stage(embed_dim * 8, embed_dim * 8, useBN=True)
        self.conv2m = add_conv_stage(embed_dim * 8, embed_dim * 4, useBN=True)
        self.conv1m = add_conv_stage(embed_dim * 4, embed_dim * 2, useBN=True)
        self.conv0m = add_conv_stage(embed_dim * 2, embed_dim, useBN=True)

        self.final_pred = nn.Conv2d(embed_dim // 2, 1, 3, 1, 1)
        self.pred0 = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        self.pred1 = nn.Conv2d(embed_dim * 2, 1, 3, 1, 1)
        self.pred2 = nn.Conv2d(embed_dim * 4, 1, 3, 1, 1)
        self.pred3 = nn.Conv2d(embed_dim * 8, 1, 3, 1, 1)
        self.fuse_pred = nn.Conv2d(3 + 5, 1, 3, 1, 1)

    def forward(self, x, nir):
        input = x
        b, c, h, w = x.shape

        # retinex
        rt_map, rt_fea = self.retinex(nir)
        x = x + x * rt_map

        # swin backbone rgb
        x = self.pebed(x)
        x = self.pos_drop(x)
        rgb_layer0, rgb_layer0_d = self.rgb_layer0(x)  # 3
        rgb_layer1, rgb_layer1_d = self.rgb_layer1(rgb_layer0_d)  # 1.5
        rgb_layer2, rgb_layer2_d = self.rgb_layer2(rgb_layer1_d)  # 0.75
        rgb_layer3 = self.rgb_layer3(rgb_layer2_d)  # 0.75

        rgb_layer0 = rgb_layer0.view(b, h // 4, w // 4, -1).permute(0, 3, 1, 2)
        rgb_layer1 = rgb_layer1.view(b, h // 8, w // 8, -1).permute(0, 3, 1, 2)
        rgb_layer2 = rgb_layer2.view(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        rgb_layer3 = rgb_layer3.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2)

        # swin backbone nir
        nir = self.pebed(nir)
        nir = self.pos_drop(nir)
        nir_layer0, nir_layer0_d = self.nir_layer0(nir)  # 3
        nir_layer1, nir_layer1_d = self.nir_layer1(nir_layer0_d)  # 1.5
        nir_layer2, nir_layer2_d = self.nir_layer2(nir_layer1_d)  # 0.75
        nir_layer3 = self.nir_layer3(nir_layer2_d)  # 0.75

        nir_layer0 = nir_layer0.view(b, h // 4, w // 4, -1).permute(0, 3, 1, 2)
        nir_layer1 = nir_layer1.view(b, h // 8, w // 8, -1).permute(0, 3, 1, 2)
        nir_layer2 = nir_layer2.view(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        nir_layer3 = nir_layer3.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2)

        # retinex feature
        rt_fea0 = self.down0(rt_fea)
        rt_fea1 = self.down1(rt_fea0)
        rt_fea2 = self.down2(rt_fea1)
        rt_fea3 = self.down3(rt_fea2)

        # mcc stage
        layer3 = self.mcc1(rgb_layer3, nir_layer3, rt_fea3)
        layer2 = self.mcc2(rgb_layer2, nir_layer2, rt_fea2)
        layer1 = self.mcc3(rgb_layer1, nir_layer1, rt_fea1)
        layer0 = self.mcc4(rgb_layer0, nir_layer0, rt_fea0)

        # Decoder
        conv3m_out = self.conv3m(layer3)  # [1, 1024, 12, 12]

        conv3m_out_ = torch.cat((self.up_32(conv3m_out), layer2), dim=1)
        conv2m_out = self.conv2m(conv3m_out_)  # [1, 512, 24, 24]

        conv2m_out_ = torch.cat((self.up_21(conv2m_out), layer1), dim=1)
        conv1m_out = self.conv1m(conv2m_out_)  # [1, 256, 48, 48]

        conv1m_out_ = torch.cat((self.up_10(conv1m_out), layer0), dim=1)
        conv0m_out = self.conv0m(conv1m_out_)  # [1, 128, 96, 96]

        convfm_out = self.up_final(conv0m_out)  # [1, 64, 384, 384]

        final_pred = self.final_pred(convfm_out)

        # Output
        layer3_pred = self.pred3(conv3m_out)
        layer2_pred = self.pred2(conv2m_out)
        layer1_pred = self.pred1(conv1m_out)
        layer0_pred = self.pred0(conv0m_out)
        layer3_pred = F.upsample(
            layer3_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer2_pred = F.upsample(
            layer2_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer1_pred = F.upsample(
            layer1_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer0_pred = F.upsample(
            layer0_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )

        # fuse predict
        fuse_feature = torch.cat(
            (input + input * rt_map, layer0_pred, layer1_pred, layer2_pred, layer3_pred, final_pred),
            dim=1,
        )
        fuse_pred = self.fuse_pred(fuse_feature)

        return (
            F.sigmoid(layer3_pred),
            F.sigmoid(layer2_pred),
            F.sigmoid(layer1_pred),
            F.sigmoid(layer0_pred),
            F.sigmoid(final_pred),
            F.sigmoid(fuse_pred),
        )

# for debug
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    net = RetinexNet().cuda()

    # print params and flops
    flops, params = profile(net, inputs=(x, x2,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    out = net(x, x2)
    print(out[0].shape)
