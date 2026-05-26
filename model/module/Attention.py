import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from thop import profile
from timm.models.layers import trunc_normal_


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.2, proj_drop=0.):
        super(WindowAttention, self).__init__()
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super(Mlp, self).__init__()
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


class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=[8, 8]):
        super(AttentionLayer, self).__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size[0]
        self.num_heads = num_heads
        # blocks
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        assert q.shape == k.shape == v.shape
        # Self Attention Stage
        # normalization
        q = q.permute(0, 2, 3, 1)  # [B H W C]
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        q = self.norm1(q)
        k = self.norm1(k)
        v = self.norm1(v)
        # window partition
        q_window = window_partition(q, self.window_size)
        k_window = window_partition(k, self.window_size)
        v_window = window_partition(v, self.window_size)
        q_window = q_window.view(-1, self.window_size * self.window_size, C)  # nW*B, win*win, C
        k_window = k_window.view(-1, self.window_size * self.window_size, C)
        v_window = v_window.view(-1, self.window_size * self.window_size, C)
        # self-attention
        feat_window = self.attn(q_window, k_window, v_window)
        # window reverse
        feat_window = feat_window.view(-1, self.window_size, self.window_size, C)  # nW*B, win, win, C
        feat = window_reverse(feat_window, self.window_size, H, W)  # B H' W' C
        feat = feat.view(B, H * W, C)
        feat = feat.transpose(1, 2).reshape(B, C, H, W)

        return feat


class SelfAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=[8, 8]):
        super(SelfAttentionTransformer, self).__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size[0]
        self.num_heads = num_heads
        # blocks
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp = Mlp(dim, hidden_features=dim*2, act_layer=nn.GELU, drop=0.2)

    def forward(self, x):
        B, C, H, W = x.shape
        # Self Attention Stage
        # normalization
        feat = x.permute(0, 2, 3, 1)  # [B H W C]
        feat = self.norm1(feat)
        # window partition
        feat_window = window_partition(feat, self.window_size)
        feat_window = feat_window.view(-1, self.window_size * self.window_size, C)  # nW*B, win*win, C
        # self-attention
        feat_window = self.attn(feat_window, feat_window, feat_window)
        # window reverse
        feat_window = feat_window.view(-1, self.window_size, self.window_size, C)  # nW*B, win, win, C
        feat = window_reverse(feat_window, self.window_size, H, W)  # B H' W' C
        feat = feat.view(B, H * W, C)
        feat = feat.transpose(1, 2).reshape(B, C, H, W)
        # shortcut
        x = x + feat
        # Channel MLP Stage
        # normalization
        feat = x.permute(0, 2, 3, 1)  # [B H W C]
        feat = self.norm2(feat)
        # mlp
        feat = self.mlp(feat)
        feat = feat.permute(0, 3, 1, 2)  # [B, C, H, W]
        # shortcut
        x = x + feat

        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=[8, 8]):
        super(CrossAttentionTransformer, self).__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size[0]
        self.num_heads = num_heads
        # blocks
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp = Mlp(dim, hidden_features=dim*2, act_layer=nn.GELU, drop=0.2)

    def forward(self, q, kv):
        B, C, H, W = q.shape
        assert q.shape == kv.shape
        # Self Attention Stage
        # normalization
        q_feat = q.permute(0, 2, 3, 1)  # [B H W C]
        kv_feat = kv.permute(0, 2, 3, 1)
        q_feat = self.norm1(q_feat)
        kv_feat = self.norm2(kv_feat)
        # window partition
        q_window = window_partition(q_feat, self.window_size)
        q_window = q_window.view(-1, self.window_size * self.window_size, C)  # nW*B, win*win, C
        kv_window = window_partition(kv_feat, self.window_size)
        kv_window = kv_window.view(-1, self.window_size * self.window_size, C)  # nW*B, win*win, C
        # cross-attention
        feat_window = self.attn(q_window, kv_window, kv_window)
        # window reverse
        feat_window = feat_window.view(-1, self.window_size, self.window_size, C)  # nW*B, win, win, C
        feat = window_reverse(feat_window, self.window_size, H, W)  # B H' W' C
        feat = feat.view(B, H * W, C)
        feat = feat.transpose(1, 2).reshape(B, C, H, W)
        # shortcut
        q = q + feat
        # Channel MLP Stage
        # normalization
        feat = q.permute(0, 2, 3, 1)  # [B H W C]
        feat = self.norm3(feat)
        # mlp
        feat = self.mlp(feat)
        feat = feat.permute(0, 3, 1, 2)  # [B, C, H, W]
        # shortcut
        q = q + feat

        return q

# for debug
if __name__ == '__main__':
    x = torch.randn(4, 768, 8, 8).cuda()
    net = CrossAttentionTransformer(768).cuda()

    flops, params = profile(net, inputs=(x, x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    result = net(x, x)
    print(result.shape)

