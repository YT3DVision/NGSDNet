import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
from torchvision.models.video.mvit import PositionalEncoding

from model.module.dsfem import DSFEM

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


class LeFF(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU):
        super(LeFF, self).__init__()

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, groups=hidden_dim,
                      kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x = self.dwconv(x) + x
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        return x


class LinearProjection2q(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,  bias=True):
        super(LinearProjection2q, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        H = int(math.sqrt(N))
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        return q  # B head N C


class LinearProjection2kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,  bias=True):
        super(LinearProjection2kv, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_kv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        H = int(math.sqrt(N))
        attn_kv = x if attn_kv is None else attn_kv
        kv = self.to_kv(attn_kv).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v, v_r1 = kv[0], kv[1], kv[2]
        return k, v  # B head N C


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


class FFM(nn.Module):
    def __init__(
            self,
            channel,
            heads,
    ):
        super(FFM, self).__init__()
        self.dim = channel
        self.heads = heads
        self.scale = self.dim // self.heads ** -0.5

        self.conv1 = SEConvBlock(channel, channel, 3)
        self.conv2 = SEConvBlock(channel, channel, 3)
        self.conv3 = SEConvBlock(channel, channel, 3)
        self.conv4 = SEConvBlock(2 * channel, channel, 3)

        self.positional_encoding1 = DSFEM(channel)
        self.positional_encoding2 = DSFEM(channel)

        self.rgb2q = LinearProjection2q(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)
        self.nir2q = LinearProjection2q(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)
        self.rgb2kv = LinearProjection2kv(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)
        self.nir2kv = LinearProjection2kv(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.norm3 = nn.LayerNorm(self.dim)
        self.norm4 = nn.LayerNorm(self.dim)

        self.mlp1 = LeFF(self.dim, self.dim)
        self.mlp2 = LeFF(self.dim, self.dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_fea, nir_fea):
        rgb = self.conv1(rgb_fea)
        nir = self.conv2(nir_fea)
        res = self.conv3(rgb - nir)

        # res to nir/rgb's kv
        res_rgb = torch.sigmoid(res) * rgb
        res_nir = torch.sigmoid(res).transpose(-2, -1) * nir

        # positional encoding
        encod_rgb = self.positional_encoding1(rgb)
        encod_nir = self.positional_encoding2(nir)

        # [B,C,H,W] -> [B,N,C]
        B, C, H, W = rgb.shape
        rgb = rgb.reshape(B, C, H * W).permute(0, 2, 1)
        nir = nir.reshape(B, C, H * W).permute(0, 2, 1)
        res_rgb = res_rgb.reshape(B, C, H * W).permute(0, 2, 1)
        res_nir = res_nir.reshape(B, C, H * W).permute(0, 2, 1)

        # normalization
        rgb = self.norm1(rgb)
        res_rgb = self.norm1(res_rgb)
        nir = self.norm2(nir)
        res_nir = self.norm2(res_nir)

        # LinearEmbedding need input [B,N,C]
        q_rgb = self.rgb2q(rgb) * self.scale
        q_nir = self.nir2q(nir) * self.scale
        k_rgb, v_rgb = self.rgb2kv(res_rgb)
        k_nir, v_nir = self.nir2kv(res_nir)

        # q k v:[B head N C]
        attn_rgb = self.softmax((q_rgb @ k_rgb.transpose(-2, -1)))
        attn_nir = self.softmax((q_nir @ k_nir.transpose(-2, -1)))

        # attn:[B head N N]
        attn_rgb = (attn_rgb @ v_rgb)
        attn_nir = (attn_nir @ v_nir)

        # attn_rgb\attn_nir: [B head N C]
        rgb = rgb.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        nir = nir.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        encod_rgb = encod_rgb.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        encod_nir = encod_nir.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn_rgb = attn_rgb + encod_rgb + rgb
        attn_nir = attn_nir + encod_nir + nir

        # [B head N C] -> [B,N,C]
        attn_rgb = attn_rgb.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_nir = attn_nir.permute(0, 2, 1, 3).reshape(B, H * W, C)

        # mlp need input [B,N,C]
        rgb = self.mlp1(self.norm3(attn_rgb)) + attn_rgb
        nir = self.mlp2(self.norm4(attn_nir)) + attn_nir

        # [B,N,C] -> [B,C,H,W]
        rgb = rgb.permute(0, 2, 1).reshape(B, C, H, W)
        nir = nir.permute(0, 2, 1).reshape(B, C, H, W)

        # CNN need input [B,C,H,W]
        fused_fea = self.conv4(torch.cat((rgb, nir), 1))
        return fused_fea
