import torch
import numpy as np
import time
import os
import math
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


class GaussianConv2d(nn.Module):
    def __init__(self, channels, kernel_size, sigma=1.0):
        super(GaussianConv2d, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.weights = self.get_gaussian_weights()

    def get_gaussian_weights(self):
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)

        return kernel

    def forward(self, x):
        self.weights = self.weights.to(x.device)
        return nn.functional.conv2d(x, self.weights, stride=1, padding=self.padding, groups=self.channels)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def gaussian_tv_loss(input_t):
    b, c, _, _ = input_t.size()
    gaussian_conv = GaussianConv2d(c,  5,  1.0)
    gaussian_input = gaussian_conv(input_t)
    tv = float(tv_loss(input_t))
    gaussian_tv = float(tv_loss(gaussian_input))
    sigma = 0.01
    e = (gaussian_tv ** 2) / sigma
    loss = tv ** 2 / math.exp(e)
    loss = torch.tensor(loss).to(input_t.device)
    return loss


def tv_loss(input_t):
    TV_Loss = TVLoss()
    return TV_Loss(input_t)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    loss = gaussian_tv_loss(x)
    print(loss)