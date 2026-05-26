import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.backbone.swin_transformer.swin_transformer_v2 import SwinTransformerV2_demo
from thop import profile
from torchvision import transforms
from model.module.Attention import CrossAttentionTransformer, SelfAttentionTransformer
from model.module.decom import CTDN
import time


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


def upsample(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            ch_coarse, ch_fine, kernel_size, stride, padding, bias=False
        ),
        nn.ReLU(),
    )


class ReflectanceGuidance(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(ReflectanceGuidance, self).__init__()
        self.dim_in = dim
        self.conv1 = SEConvBlock(dim, dim, 3, 1, 1)
        self.cr1 = BasicConv2d(2*dim, dim, 1, 1, 0)
        self.cr2 = BasicConv2d(dim, 1, 3, 1, 1)
        self.SA = SelfAttentionTransformer(dim, num_heads, [12, 12])

    def forward(self, rgb, nir, mask):
        # input size [B, N, C] -> [B, C, H, W]
        b, n, c = nir.shape
        h = w = int(math.sqrt(n))
        nir = nir.view(b, h, w, -1).permute(0, 3, 1, 2) # X^i_n

        # guidance stage
        mask = torch.sigmoid(self.cr2(mask)) # M_c
        map = torch.sigmoid(self.conv1(nir-rgb)) # M_d
        fusion = self.cr1(torch.cat([nir, rgb], dim=1)) # X^i_f
        fusion = self.SA(fusion * map * mask)
        nir = nir + fusion  # hat{X}^i_n

        # output size [B, C, H, W] -> [B, N, C]
        nir = nir.permute(0, 2, 3, 1).view(b, n, -1)
        return nir


class IlluminationGatedFusion(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(IlluminationGatedFusion, self).__init__()
        self.dim_in = dim
        self.attn1 = CrossAttentionTransformer(dim, num_heads, window_size)
        self.attn2 = CrossAttentionTransformer(dim, num_heads, window_size)
        self.attn3 = CrossAttentionTransformer(dim, num_heads, window_size)
        self.attn4 = CrossAttentionTransformer(dim, num_heads, window_size)

    def forward(self, x, rgb, nir, wr, wn):
        input_size = x.shape[-1]
        wr = wr.unsqueeze(1)
        wn = wn.unsqueeze(1)
        wr = F.interpolate(wr, size=(input_size, input_size), mode='bilinear', align_corners=True)
        wn = F.interpolate(wn, size=(input_size, input_size), mode='bilinear', align_corners=True)

        rgb = self.attn1(nir, rgb)
        nir = self.attn2(rgb, nir)
        rgb = self.attn3(x, rgb)
        nir = self.attn4(x, nir)

        return rgb * wr + nir * wn


class SpatialAttention(nn.Module):
    # modified from CBAM spatial attention
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class IlluminationEstimate(nn.Module):
    def __init__(self, input_size):
        super(IlluminationEstimate, self).__init__()
        self.mid_size = input_size // 4
        self.mid_dim = 16
        self.gap = nn.AdaptiveAvgPool2d((self.mid_size, self.mid_size))
        self.conv1 = DepthWiseConv2d(1, self.mid_dim, 5, 1, 2)
        self.conv2 = DepthWiseConv2d(1, self.mid_dim, 5, 1, 2)
        self.cr1 = BasicConv2d(2 * self.mid_dim, self.mid_dim, 3, 1, 1)
        self.cr2 = BasicConv2d(2 * self.mid_dim, self.mid_dim, 3, 1, 1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.toweight1 = BasicConv2d(self.mid_dim, 4, 1, 1, 0)
        self.toweight2 = BasicConv2d(self.mid_dim, 4, 1, 1, 0)

    def forward(self, illu, illu2):
        illu = self.gap(illu)
        illu2 = self.gap(illu2)
        illu = self.conv1(illu)
        illu2 = self.conv2(illu2)

        map1 = self.cr1(torch.cat((illu, illu - illu2), dim=1))
        map2 = self.cr2(torch.cat((illu2, illu2 - illu), dim=1))
        wr = torch.sigmoid(self.toweight1(self.sa1(map1) * map1))
        wn = torch.sigmoid(self.toweight1(self.sa2(map2) * map2))

        return wr, wn


class RetinexDecomposition(nn.Module):
    def __init__(self, stage1_path=None):
        super(RetinexDecomposition, self).__init__()
        ctdn = CTDN()
        if stage1_path is not None:
            state_dict = torch.load(stage1_path)
            pretrained_dict = state_dict["model"]
            ctdn.load_state_dict(pretrained_dict, strict=False)
        self.ReconNet = ctdn.ReconNet
        self.retinex = ctdn.retinex

    def forward(self, x):
        features = self.ReconNet(x, pred_fea=None)
        R, L = self.retinex(features)
        R = self.ReconNet(x, pred_fea=R)
        L = F.interpolate(L, scale_factor=2, mode='bilinear', align_corners=True)
        return R, L


class NIRNet(nn.Module):
    def __init__(self, backbone_path=None, stage1_path=None):
        super(NIRNet, self).__init__()
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
        # backbone的参数
        embed_dim = 128
        img_size = 384

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

        # Retinex Decomposition
        self.decom0 = RetinexDecomposition(stage1_path)
        self.decom1 = RetinexDecomposition(stage1_path)

        # Illumination Estimate
        self.illu_estimator = IlluminationEstimate(img_size)

        #####  Reflectance Guidance  #####
        self.reflectance_guidance0 = ReflectanceGuidance(embed_dim, 4)
        self.reflectance_guidance1 = ReflectanceGuidance(embed_dim * 2, 8)
        self.reflectance_guidance2 = ReflectanceGuidance(embed_dim * 4, 16)
        self.reflectance_guidance3 = ReflectanceGuidance(embed_dim * 8, 32)
        # Reflectance Mask Upsample
        self.mask_32 = upsample(embed_dim * 8, embed_dim * 4)
        self.mask_21 = upsample(embed_dim * 4, embed_dim * 2)
        self.mask_10 = upsample(embed_dim * 2, embed_dim)
        # Mask Refinement
        self.mask_refine = SEConvBlock(embed_dim * 8, embed_dim * 8, 3, 1, 1)

        # channel reduction
        self.cr0 = BasicConv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.cr1 = BasicConv2d(embed_dim * 4, embed_dim * 2, 1, 1, 0)
        self.cr2 = BasicConv2d(embed_dim * 8, embed_dim * 4, 1, 1, 0)
        self.cr3 = BasicConv2d(embed_dim * 16, embed_dim * 8, 1, 1, 0)
        # bottom
        self.fusion = BasicConv2d(embed_dim * 16, embed_dim * 8, 1, 1, 0)

        #####  Decoder Stage  #####
        self.fusion3 = IlluminationGatedFusion(embed_dim * 8, 4, [12, 12])
        self.fusion2 = IlluminationGatedFusion(embed_dim * 4, 8, [12, 12])
        self.fusion1 = IlluminationGatedFusion(embed_dim * 2, 16, [12, 12])
        self.fusion0 = IlluminationGatedFusion(embed_dim, 32, [12, 12])
        # decoder upsample
        self.up_32 = upsample(embed_dim * 8, embed_dim * 4)
        self.up_21 = upsample(embed_dim * 4, embed_dim * 2)
        self.up_10 = upsample(embed_dim * 2, embed_dim)
        self.up_final = upsample(embed_dim, embed_dim // 2, kernel_size=8, stride=4, padding=2)

        # predict conv
        self.final_pred = nn.Conv2d(embed_dim // 2, 1, 3, 1, 1)
        self.pred0 = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        self.pred1 = nn.Conv2d(embed_dim * 2, 1, 3, 1, 1)
        self.pred2 = nn.Conv2d(embed_dim * 4, 1, 3, 1, 1)
        self.pred3 = nn.Conv2d(embed_dim * 8, 1, 3, 1, 1)
        self.fuse_pred = nn.Conv2d(3 + 5, 1, 3, 1, 1)

        # auxi supervise
        self.auxi0 = nn.Conv2d(embed_dim * 8, 1, 3, 1, 1)
        self.auxi1 = nn.Conv2d(embed_dim * 8, 1, 3, 1, 1)
        self.edge0 = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        self.edge1 = nn.Conv2d(embed_dim * 2, 1, 3, 1, 1)
        self.edge2 = nn.Conv2d(embed_dim * 4, 1, 3, 1, 1)
        self.edge3 = nn.Conv2d(embed_dim * 8, 1, 3, 1, 1)

    def forward(self, x, nir):
        input = x
        b, c, h, w = x.shape
        
        # for test
        # nir = torch.mean(x, dim=1, keepdim=True)
        # nir = torch.cat((nir, nir, nir), dim=1)
        
        # Retinex Decomposition
        refl, illu = self.decom0(x)
        refl2, illu2 = self.decom0(nir)
        # Illumination Estimate
        wr, wn = self.illu_estimator(illu, illu2)
        wr = wr.transpose(0, 1)  # [c, b, h//4, w//4]
        wn = wn.transpose(0, 1)  # [c, b, h//4, w//4]

        # backbone for rgb feature
        x = refl
        x = self.pebed(x)
        x = self.pos_drop(x)
        rgb_layer0, rgb_layer0_d = self.rgb_layer0(x)
        rgb_layer1, rgb_layer1_d = self.rgb_layer1(rgb_layer0_d)
        rgb_layer2, rgb_layer2_d = self.rgb_layer2(rgb_layer1_d)
        rgb_layer3 = self.rgb_layer3(rgb_layer2_d)

        # reshape backbone output
        rgb_layer0 = rgb_layer0.view(b, h // 4, w // 4, -1).permute(0, 3, 1, 2)
        rgb_layer1 = rgb_layer1.view(b, h // 8, w // 8, -1).permute(0, 3, 1, 2)
        rgb_layer2 = rgb_layer2.view(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        rgb_layer3 = rgb_layer3.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2)

        # backbone for nir feature
        nir = refl2
        nir = self.pebed(nir)
        nir = self.pos_drop(nir)

        # Reflectance Guidance Encoder Stage
        # Reflectance Mask Upsample
        mask3 = rgb_layer3 + self.mask_refine(rgb_layer3)
        mask2 = self.mask_32(mask3)
        mask1 = self.mask_21(mask2)
        mask0 = self.mask_10(mask1)

        # Reflectance Guidance
        # [b, c, h // 4, w // 4]
        nir_layerx_d = self.reflectance_guidance0(rgb_layer0, nir, mask0)
        nir_layer0, nir_layer0_d = self.nir_layer0(nir_layerx_d)
        # [b, 2c, h // 8, w // 8]
        nir_layer0_d = self.reflectance_guidance1(rgb_layer1, nir_layer0_d, mask1)
        nir_layer1, nir_layer1_d = self.nir_layer1(nir_layer0_d)
        # [b, 4c, h // 16, w // 16]
        nir_layer1_d = self.reflectance_guidance2(rgb_layer2, nir_layer1_d, mask2)
        nir_layer2, nir_layer2_d = self.nir_layer2(nir_layer1_d)
        # [b, 8c, h // 32, w // 32]
        nir_layer2_d = self.reflectance_guidance3(rgb_layer3, nir_layer2_d, mask3)
        nir_layer3 = self.nir_layer3(nir_layer2_d)

        # reshape backbone output
        nir_layer0 = nir_layer0.view(b, h // 4, w // 4, -1).permute(0, 3, 1, 2)
        nir_layer1 = nir_layer1.view(b, h // 8, w // 8, -1).permute(0, 3, 1, 2)
        nir_layer2 = nir_layer2.view(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        nir_layer3 = nir_layer3.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2)

        # backward gradient is needed
        nir_layerx_d = nir_layerx_d.view(b, h // 4, w // 4, -1).permute(0, 3, 1, 2)
        nir_layer0_d = nir_layer0_d.view(b, h // 8, w // 8, -1).permute(0, 3, 1, 2)
        nir_layer1_d = nir_layer1_d.view(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        nir_layer2_d = nir_layer2_d.view(b, h // 32, w // 32, -1).permute(0, 3, 1, 2)

        # channel reduction
        nir_layer0 = self.cr0(torch.cat([nir_layer0, nir_layerx_d], dim=1))
        nir_layer1 = self.cr1(torch.cat([nir_layer1, nir_layer0_d], dim=1))
        nir_layer2 = self.cr2(torch.cat([nir_layer2, nir_layer1_d], dim=1))
        nir_layer3 = self.cr3(torch.cat([nir_layer3, nir_layer2_d], dim=1))

        # decoder stage
        semantic = self.fusion(torch.cat([rgb_layer3, nir_layer3], 1))
        layer3 = self.fusion3(semantic, rgb_layer3, nir_layer3, wr[3], wn[3])
        layer2 = self.fusion2(self.up_32(layer3), rgb_layer2, nir_layer2, wr[2], wn[2])
        layer1 = self.fusion1(self.up_21(layer2), rgb_layer1, nir_layer1, wr[1], wn[1])
        layer0 = self.fusion0(self.up_10(layer1), rgb_layer0, nir_layer0, wr[0], wn[0])
        final_out = self.up_final(layer0)

        # predict stage
        final_pred = self.final_pred(final_out)
        layer3_pred = self.pred3(layer3)
        layer2_pred = self.pred2(layer2)
        layer1_pred = self.pred1(layer1)
        layer0_pred = self.pred0(layer0)

        layer3_pred = F.upsample(layer3_pred, size=input.size()[2:], mode="bilinear", align_corners=True)
        layer2_pred = F.upsample(layer2_pred, size=input.size()[2:], mode="bilinear", align_corners=True)
        layer1_pred = F.upsample(layer1_pred, size=input.size()[2:], mode="bilinear", align_corners=True)
        layer0_pred = F.upsample(layer0_pred, size=input.size()[2:], mode="bilinear", align_corners=True)

        # for training
        if self.training:
            auxi0_pred = self.auxi0(mask3)
            auxi1_pred = self.auxi1(nir_layer3)
            edge0 = self.edge0(layer0)
            edge1 = self.edge1(layer1)
            edge2 = self.edge2(layer2)
            edge3 = self.edge3(layer3)
            auxi0_pred = F.upsample(auxi0_pred, size=input.size()[2:], mode="bilinear", align_corners=True)
            auxi1_pred = F.upsample(auxi1_pred, size=input.size()[2:], mode="bilinear", align_corners=True)
            edge0 = F.upsample(edge0, size=input.size()[2:], mode="bilinear", align_corners=True)
            edge1 = F.upsample(edge1, size=input.size()[2:], mode="bilinear", align_corners=True)
            edge2 = F.upsample(edge2, size=input.size()[2:], mode="bilinear", align_corners=True)
            edge3 = F.upsample(edge3, size=input.size()[2:], mode="bilinear", align_corners=True)

        fuse_feature = torch.cat((input, layer0_pred, layer1_pred, layer2_pred, layer3_pred, final_pred), 1)
        fuse_pred = self.fuse_pred(fuse_feature)

        if self.training:
            return (
                torch.sigmoid(layer3_pred),
                torch.sigmoid(layer2_pred),
                torch.sigmoid(layer1_pred),
                torch.sigmoid(layer0_pred),
                torch.sigmoid(final_pred),
                torch.sigmoid(fuse_pred),
                torch.sigmoid(auxi0_pred),
                torch.sigmoid(auxi1_pred),
                torch.sigmoid(edge0),
                torch.sigmoid(edge1),
                torch.sigmoid(edge2),
                torch.sigmoid(edge3),
            )
        else:
            return torch.sigmoid(fuse_pred)


# for debug
if __name__ == "__main__":
    x1 = torch.randn(1, 3, 384, 384).cuda()
    x2 = torch.randn(1, 3, 384, 384).cuda()
    net = NIRNet(
        #backbone_path="../model/backbone/swin_transformer/swinv2_base_patch4_window24_384.pth",
        #stage1_path="../ckpt/stage1_weight.pth",
                 ).cuda().eval()

    with torch.no_grad():
        # print params and flops
        flops, params = profile(net, inputs=(x1, x2))
        print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
        print("params=", str(params / 1e6) + '{}'.format("M"))

        total = sum(p.numel() for p in net.parameters())
        print("Total params: %.2fM" % (total / 1e6))

        t = time.time()
        out = net(x1, x2)
        print('test time : %.3fs' % (time.time() - t))
        print(out[0].shape)

