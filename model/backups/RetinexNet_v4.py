import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.backbone.swin_transformer.swin_transformer_v2 import SwinTransformerV2_demo
from thop import profile
from torchvision import transforms

import time
from PIL import Image
import os

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


class SEConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction=16):
        super(SEConvBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = DepthWiseConv2d(dim, dim, kernel_size, 1, self.padding)
        self.se = SElayer(dim, reduction)

    def forward(self, x):
        return x + self.se(self.conv(x))


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


class CrossPool(nn.Module):
    def __init__(self, dim):
        super(CrossPool, self).__init__()
        self.dim = dim
        self.se1 = SEConvBlock(dim, 3)
        self.se2 = SEConvBlock(dim, 3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.fusion = BasicConv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, hf, lf):
        hf = self.se1(hf)
        lf = self.se2(lf)
        hf_p = self.maxpool(hf)
        lf_p = self.avgpool(lf)
        hf_p = F.interpolate(hf_p, scale_factor=2, mode='bilinear', align_corners=True)
        lf_p = F.interpolate(lf_p, scale_factor=2, mode='bilinear', align_corners=True)
        map = torch.sigmoid(hf_p - lf_p)
        hf = map * hf
        lf = map * lf
        result = self.fusion(torch.cat((hf, lf), 1))
        return result


class RetinexBlock(nn.Module):
    def __init__(self, dim_middle):
        super(RetinexBlock, self).__init__()
        self.dim_middle = dim_middle

        self.conv1 = DepthWiseConv2d(4, dim_middle, 3, 1, 1)
        self.conv2 = DepthWiseConv2d(dim_middle, dim_middle, 5, 1, 2)
        self.conv3 = DepthWiseConv2d(dim_middle // 4 * 3, 3, 3, 1, 1)
        self.conv4 = DepthWiseConv2d(dim_middle // 4, 1, 3, 1, 1)

    def forward(self, x):
        mean_c = x.mean(dim=1).unsqueeze(1)
        x = torch.cat([x, mean_c], dim=1)
        x = self.conv1(x)
        rt_fea = self.conv2(x)
        R_fea = rt_fea[:, 0:self.dim_middle // 4 * 3, :, :]
        I_fea = rt_fea[:, self.dim_middle // 4 * 3:self.dim_middle, :, :]
        R = self.conv3(R_fea)
        I = self.conv4(I_fea)
        return R_fea, R, I


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


class RetinexNet(nn.Module):
    def __init__(
        self,
        backbone_path=None,
    ):
        super(RetinexNet, self).__init__()
        swin_transformer = SwinTransformerV2_demo(
            img_size=384,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            window_size=24,
        )

        swin_transformer2 = SwinTransformerV2_demo(
            img_size=384,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            window_size=24,
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
        embed_dim = 128

        # retinex
        self.retinex = RetinexBlock(embed_dim // 4)
        self.down0 = downsample(embed_dim // 16 * 3, embed_dim, 8, 4, 2)
        self.down1 = downsample(embed_dim, embed_dim * 2)
        self.down2 = downsample(embed_dim * 2, embed_dim * 4)
        self.down3 = downsample(embed_dim * 4, embed_dim * 8)

        # feature extract
        self.fe_rgb0 = CrossPool(embed_dim)
        self.fe_rgb1 = CrossPool(embed_dim * 2)
        self.fe_rgb2 = CrossPool(embed_dim * 4)
        self.fe_rgb3 = CrossPool(embed_dim * 8)

        self.fe_nir0 = CrossPool(embed_dim)
        self.fe_nir1 = CrossPool(embed_dim * 2)
        self.fe_nir2 = CrossPool(embed_dim * 4)
        self.fe_nir3 = CrossPool(embed_dim * 8)

        # mcc
        self.fusion1 = SKFF(embed_dim * 8, 2)
        self.fusion2 = SKFF(embed_dim * 4, 2)
        self.fusion3 = SKFF(embed_dim * 2, 2)
        self.fusion4 = SKFF(embed_dim, 2)

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
        # self.fuse_pred = nn.Conv2d(3 + 5, 1, 3, 1, 1)
        self.fuse_pred = SKFF(1, 8)

    def forward(self, x, nir):
        input = x
        b, c, h, w = x.shape

        # retinex
        rt_fea, R_map, I_map = self.retinex(x)

        # save lightened
        # lighten = R_map.data.squeeze(0)
        # lighten = np.array(transforms.Resize((512, 512))(to_pil(lighten)))
        # Image.fromarray(lighten).save(os.path.join('C:/Users/32319/Desktop/Low-Light Glass Detection/MyGlassNet/output', str(int(time.time())) + ".png"))

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

        # feature extract
        rgb_layer0 = self.fe_rgb0(rgb_layer0, rt_fea0)
        rgb_layer1 = self.fe_rgb1(rgb_layer1, rt_fea1)
        rgb_layer2 = self.fe_rgb2(rgb_layer2, rt_fea2)
        rgb_layer3 = self.fe_rgb3(rgb_layer3, rt_fea3)

        nir_layer0 = self.fe_nir0(nir_layer0, rt_fea0)
        nir_layer1 = self.fe_nir1(nir_layer1, rt_fea1)
        nir_layer2 = self.fe_nir2(nir_layer2, rt_fea2)
        nir_layer3 = self.fe_nir3(nir_layer3, rt_fea3)

        # mcc stage
        layer3 = self.fusion1([rgb_layer3, nir_layer3])
        layer2 = self.fusion2([rgb_layer2, nir_layer2])
        layer1 = self.fusion3([rgb_layer1, nir_layer1])
        layer0 = self.fusion4([rgb_layer0, nir_layer0])

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
        """
        fuse_feature = torch.cat(
            (input, layer0_pred, layer1_pred, layer2_pred, layer3_pred, final_pred),
            dim=1,
        )
        fuse_pred = self.fuse_pred(fuse_feature)
        """
        fuse_pred = self.fuse_pred([input[:, 0:1, :, :], input[:, 1:2, :, :], input[:, 2:3, :, :],
                                    layer0_pred, layer1_pred, layer2_pred, layer3_pred, final_pred])

        return (
            F.sigmoid(layer3_pred),
            F.sigmoid(layer2_pred),
            F.sigmoid(layer1_pred),
            F.sigmoid(layer0_pred),
            F.sigmoid(final_pred),
            F.sigmoid(fuse_pred),
            R_map,
            I_map
        )


# for debug
if __name__ == "__main__":
    x = torch.randn(1, 3, 384, 384).cuda()
    x2 = torch.randn(1, 3, 384, 384).cuda()
    net = RetinexNet(backbone_path='./backbone/swin_transformer/swinv2_base_patch4_window24_384.pth').cuda()

    # print params and flops
    flops, params = profile(net, inputs=(x, x2,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    out = net(x, x2)
    print(out[0].shape)
