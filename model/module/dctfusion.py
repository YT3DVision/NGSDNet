import torch
import torch.nn as nn

from model.module.dct import dct_layer, reverse_dct_layer


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
    def __init__(self, dim_in, dim_out, kernel_size):
        super(DepthWiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
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
    def __init__(self, dim_in, dim_out, kernel):
        super(SEConvBlock, self).__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel_size=kernel)
        self.se = SElayer(dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x) + x
        return x


class DCT_Fusion(nn.Module):
    def __init__(self, channel):
        super(DCT_Fusion, self).__init__()
        self.dim = channel
        self.dct_kernel = 8

        self.conv1 = SEConvBlock(self.dim, self.dim, 3)
        self.conv2 = SEConvBlock(self.dim, self.dim, 3)
        self.conv3 = SEConvBlock(self.dim, self.dim, 3)
        self.conv4 = SEConvBlock(self.dim, self.dim, 3)
        self.conv5 = SEConvBlock(self.dim, self.dim, 3)

        self.dct1 = dct_layer(self.dim, self.dct_kernel, self.dct_kernel)
        self.dct2 = dct_layer(self.dim, self.dct_kernel, self.dct_kernel)
        self.idct = reverse_dct_layer(self.dim, self.dct_kernel, self.dct_kernel)

        self.reduction = SEConvBlock(self.dim * 2, self.dim, 3)

    def forward(self, rgb, nir):
        # parameter
        B, C, H, W = rgb.shape
        k = self.dct_kernel

        # spatial feature
        rgb_s = self.conv1(rgb)
        nir_s = self.conv2(nir)

        # frequency feature
        rgb_f = self.dct1(rgb)  # [B, C*k*k, H//k, W//k]
        nir_f = self.dct2(nir)

        rgb_f = rgb_f.reshape(B, -1, k, k, H//k, W//k).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, H, W)
        nir_f = nir_f.reshape(B, -1, k, k, H//k, W//k).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, H, W)

        rgb_f = self.conv3(rgb_f)  # [B, C, H, W]
        nir_f = self.conv4(nir_f)

        # residual feature
        residual = self.conv5(rgb_f - nir_f)  # [B, C, H, W]
        residual = residual.reshape(B, C, H // k, k, W // k, k).permute(0, 1, 3, 5, 2, 4).contiguous()
        residual = residual.reshape(B, -1, H // k, W // k)  # [B, C*k*k, H//k, W//k]

        residual = self.idct(residual)  # [B, C, H, W]
        residual = torch.sigmoid(residual)

        # feature fusion
        out = torch.cat((rgb_s, nir_s), dim=1)
        out = self.reduction(out)
        out = out * residual

        return out


# for test
if __name__ == "__main__":
    rgb = torch.randn(1, 4, 16, 16)
    nir = torch.randn(1, 4, 16, 16)
    b, c, h, w = rgb.shape

    model = DCT_Fusion(c)
    out = model(rgb, nir)
    print(out)
