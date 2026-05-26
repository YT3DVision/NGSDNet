import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np

from model.module.cbam import CBAM

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
        """
        input
            x: B L C
        output
            x: B L C
        """
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
        # self.LN1 = nn.LayerNorm(dim_in)
        self.act1 = nn.ReLU()
        self.point_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        # self.LN2 = nn.LayerNorm(dim_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        # x = self.LN1(x)
        x = self.act1(x)
        x = self.point_conv(x)
        # x = self.LN2(x)
        out = self.act2(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention, self).__init__()
        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.se_module(x)
        return x


class RCAB(nn.Module):
    def __init__(self, in_channels, reduction):
        super(RCAB, self).__init__()
        self.rcab = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            ChannelAttention(in_channels, reduction)
        )

    def forward(self, x):
        return x + self.rcab(x)


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

        self.CBAM1 = CBAM(channel)
        self.CBAM2 = CBAM(channel)
        self.CBAM3 = CBAM(channel)
        self.CBAM4 = CBAM(channel)

        self.conv_fusion = DepthWiseConv2d(2 * channel, channel, 3)

        self.RCAB1 = RCAB(in_channels=self.dim, reduction=1)
        self.RCAB2 = RCAB(in_channels=self.dim * 2, reduction=1)

        self.rgb2q = LinearProjection2q(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)
        self.nir2q = LinearProjection2q(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)
        self.resi2kv = LinearProjection2kv(dim=self.dim, heads=self.heads, dim_head=self.dim // self.heads)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp1 = LeFF(self.dim, self.dim)
        self.mlp2 = LeFF(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_fea, nir_fea):
        rgb = self.CBAM1(rgb_fea)
        nir = self.CBAM2(nir_fea)
        res = self.RCAB1(self.CBAM3(rgb_fea)-self.CBAM4(nir_fea))

        # [B,C,H,W] -> [B,N,C]
        B, C, H, W = rgb.shape
        rgb = rgb.reshape(B, C, H * W).permute(0, 2, 1)
        nir = nir.reshape(B, C, H * W).permute(0, 2, 1)
        res = res.reshape(B, C, H * W).permute(0, 2, 1)

        # LinearEmbedding need input [B,N,C]
        q_rgb = self.rgb2q(rgb) * self.scale
        q_nir = self.nir2q(nir) * self.scale
        k, v = self.resi2kv(res)

        # q k v:[B head N C]
        attn_rgb = self.softmax((q_rgb @ k.transpose(-2, -1)))
        attn_nir = self.softmax((q_nir @ k.transpose(-2, -1)))

        # attn:[B head N N]
        attn_rgb = (attn_rgb @ v)
        attn_nir = (attn_nir @ v)

        # attn_rgb\attn_nir: [B head N C]
        rgb = rgb.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        nir = nir.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attn_rgb = attn_rgb + rgb
        attn_nir = attn_nir + nir

        # [B head N C] -> [B,N,C]
        attn_rgb = attn_rgb.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_nir = attn_nir.permute(0, 2, 1, 3).reshape(B, H * W, C)

        # mlp need input [B,N,C]
        rgb = self.mlp1(self.norm1(attn_rgb)) + attn_rgb
        nir = self.mlp2(self.norm2(attn_nir)) + attn_nir

        # [B,N,C] -> [B,C,H,W]
        rgb = rgb.permute(0, 2, 1).reshape(B, C, H, W)
        nir = nir.permute(0, 2, 1).reshape(B, C, H, W)

        # CNN need input [B,C,H,W]
        fused_fea = self.RCAB2(torch.cat((rgb, nir), 1))
        fused_fea = self.conv_fusion(fused_fea)
        return fused_fea
