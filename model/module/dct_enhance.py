import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from model.module.mirror_head import SFA, Contextcontrast

class DCT_Enhance(nn.Module):
    def __init__(self, in_channels, input_size, num_head=8, dct_kernel=(8, 8)):
        super(DCT_Enhance, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_head = num_head
        self.dct_kernel = dct_kernel

        self.sfa = SFA(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)
        self.context = Contextcontrast(dct_kernel[0], dct_kernel[1], input_size, num_head, in_channels)

    def forward(self, x):
        y = self.sfa(x)
        result = self.context(x, y)

        return result

# for debug
if __name__ == '__main__':
    img = torch.rand(1, 8, 16, 16)
    model = DCT_Enhance(in_channels=8, input_size=16, num_head=8)
    out = model(img)
    print(out)

