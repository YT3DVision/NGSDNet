import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from thop import profile

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.mmseg.shape_convert import nchw_to_nlc, nlc_to_nchw
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

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = self.q(q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SFA(nn.Module):
    """
    RGB_imgs, (n, c, h, w)

    """

    def __init__(self, dct_kernel=8, window_size=8, input_size=16, num_heads=8, dim=1024, mlp_ratio=2., drop_path=0.):
        super(SFA, self).__init__()
        self.dct_kernel = dct_kernel
        self.window_size = window_size
        self.num_heads = num_heads
        self.output_size = math.ceil(input_size / dct_kernel) * dct_kernel

        self.dct = dct_layer(in_c=dim, h=dct_kernel, w=dct_kernel)
        self.rdct = reverse_dct_layer(out_c=dim, h=dct_kernel, w=dct_kernel)
        self.fold = nn.Fold(output_size=(self.output_size, self.output_size), kernel_size=(dct_kernel, dct_kernel),
                            stride=dct_kernel)
        self.unfold = nn.Unfold(kernel_size=(dct_kernel, dct_kernel), stride=dct_kernel)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.mask = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.attn1 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)

        self.attn2 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)

        # self.enhanced = EnhancedAttention(dim=dim, num_heads=self.num_heads)
        self.concat = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, feat, paddings=(0, 0)):
        B, C, H, W = feat.shape

        padding_h, padding_w = paddings
        assert padding_h == 0 and padding_w == 0

        x_spac = feat

        # dct
        x_freq = self.dct(feat)  # B C*64 H/8 W/8
        B_f, C_f, H_f, W_f = x_freq.shape
        x_freq = x_freq.flatten(2)  # B C*64 H/8*W/8
        x_freq = self.fold(x_freq)  # B C H W

        # mask
        x_freq = self.mask(x_freq)

        x_freq = self.norm1(x_freq)
        x_spac = self.norm2(x_spac)

        x_freq = x_freq.permute(0, 2, 3, 1)
        x_spac = x_spac.permute(0, 2, 3, 1)

        # partition windows
        x_freq_windows = window_partition(x_freq, self.window_size)  # nW*B, window_size, window_size, C
        x_freq_windows = x_freq_windows.view(-1, self.window_size * self.window_size,
                                             C)  # nW*B, window_size*window_size, C
        x_spac_windows = window_partition(x_spac, self.window_size)  # nW*B, window_size, window_size, C
        x_spac_windows = x_spac_windows.view(-1, self.window_size * self.window_size,
                                             C)  # nW*B, window_size*window_size, C

        # space to frequency
        stf_windows = self.attn1(x_spac_windows, x_freq_windows, x_freq_windows)  # nW*B, window_size*window_size, C

        # merge windows
        stf_windows = stf_windows.view(-1, self.window_size, self.window_size, C)
        stf = window_reverse(stf_windows, self.window_size, H, W)  # B H' W' C
        stf = stf.view(B, H * W, C)

        # rdct
        stf = stf.transpose(1, 2).reshape(B, C, H, W)
        stf = self.unfold(stf)  # B C*64 H/8*W/8
        stf = stf.view(B_f, C_f, H_f, W_f)
        stf = self.rdct(stf)  # B C H W

        # frequency to space
        fts_windows = self.attn2(x_freq_windows, x_spac_windows, x_spac_windows)

        # merge windows
        fts_windows = fts_windows.view(-1, self.window_size, self.window_size, C)
        fts = window_reverse(fts_windows, self.window_size, H, W)  # B H' W' C
        fts = fts.view(B, H * W, C)
        fts = fts.transpose(1, 2).reshape(B, C, H, W)

        # fusion
        # out = self.enhanced(stf, fts, feat)
        out = torch.cat((stf, fts), dim=1)
        out = self.concat(out)

        # residual
        shortcut = feat
        # shortcut = remove_image_padding(shortcut, padding_h, padding_w)

        out = shortcut + self.drop_path(out)
        out = nchw_to_nlc(out)  # B H*W C
        out = out + self.drop_path(self.mlp(self.norm3(out)))  # B, H * W, C
        out = nlc_to_nchw(out, [H, W])

        return out
class DWA(nn.Module):
    def __init__(self, dct_kernel=8, window_size=8, input_size=16, num_heads=8, dim=1024, mlp_ratio=2., drop_path=0.):
        super(DWA, self).__init__()
        self.dct_kernel = dct_kernel
        self.window_size = window_size
        self.num_heads = num_heads

        self.u1 = nn.Unfold(kernel_size=self.dct_kernel, dilation=input_size // self.dct_kernel, padding=0, stride=1)
        self.f1 = nn.Fold(output_size=input_size, kernel_size=self.dct_kernel, dilation=1, padding=0,
                          stride=self.dct_kernel)
        self.norm1 = LayerNorm2d(dim)
        self.re_u = nn.Unfold(kernel_size=self.dct_kernel, dilation=1, padding=0, stride=self.dct_kernel)
        self.re_f = nn.Fold(output_size=input_size, kernel_size=self.dct_kernel, dilation=input_size // self.dct_kernel,
                            padding=0, stride=1)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=True, qk_scale=None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.norm3 = LayerNorm2d(dim)
        self.act = nn.GELU()

        self.concat = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, feat):
        B, C, H, W = feat.shape
        dilated = self.f1(self.u1(feat))
        dilated = self.norm1(dilated)
        dilated = dilated.permute(0, 2, 3, 1)
        dilated_window = window_partition(dilated, self.window_size)
        dilated_window = dilated_window.view(-1, self.window_size * self.window_size,
                                             C)  # nW*B, window_size*window_size, C

        context_windows = self.attn(dilated_window, dilated_window, dilated_window)

        context_windows = context_windows.view(-1, self.window_size, self.window_size, C)
        context = window_reverse(context_windows, self.window_size, H, W)  # B H' W' C
        context = context.view(B, H * W, C)
        context = context.transpose(1, 2).reshape(B, C, H, W)

        context = self.re_f(self.re_u(context))

        # residual
        shortcut = feat
        # shortcut = remove_image_padding(shortcut, padding_h, padding_w)

        context = shortcut + self.drop_path(context)
        context = nchw_to_nlc(context)  # B H*W C
        context = context + self.drop_path(self.mlp(self.norm2(context)))  # B, H * W, C
        context = nlc_to_nchw(context, [H, W])

        return context

class DCT_Fusion(nn.Module):
    def __init__(self, in_channels, input_size, num_head=8, dct_kernel=(8, 8)):
        super(DCT_Fusion, self).__init__()
        # parameters
        self.dim_in = in_channels
        self.input_size = input_size
        self.num_head = num_head
        self.dct_kernel = dct_kernel
        self.window_size = dct_kernel[1]

        # blocks
        self.sfa1 = SFA(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)
        self.sfa2 = SFA(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)
        self.dwa1 = DWA(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)
        self.dwa2 = DWA(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)

        self.cr = SEConvBlock(2 * in_channels, in_channels, 3)

    def forward(self, rgb, nir):
        rgb_f = self.sfa1(rgb)
        nir_f = self.sfa2(nir)
        rgb_s = self.dwa1(rgb)
        nir_s = self.dwa2(nir)

        map = torch.sigmoid(rgb_f - nir_f)
        rgb_residual = map * rgb_s - (1 - map) * rgb_f
        nir_residual = map * nir_s - (1 - map) * nir_f
        result = torch.cat((rgb_residual, nir_residual), dim=1)
        result = self.cr(result)

        return result


# for debug
if __name__ == "__main__":
    rgb = torch.randn(1, 96, 64, 64).cuda()
    nir = torch.randn(1, 96, 64, 64).cuda()
    b, c, h, w = rgb.shape

    model = DCT_Fusion(c, h, 3, (8, 8)).cuda()
    flops, params = profile(model, inputs=(rgb, nir,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    out = model(rgb, nir)
    print(out.shape)
