import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from thop import profile
from torchvision import transforms
from model.module.decom import CTDN


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


class RetinexNet(nn.Module):
    def __init__(self, stage1_path=None):
        super(RetinexNet, self).__init__()
        # Retinex Decomposition
        self.decom = RetinexDecomposition(stage1_path)

    def forward(self, x, nir):
        # Retinex Decomposition
        refl, illu = self.decom(x)
        refl2, illu2 = self.decom(nir)
        return refl, illu, refl2, illu2

# for debug
if __name__ == "__main__":
    x1 = torch.randn(2, 3, 384, 384).cuda()
    x2 = torch.randn(2, 3, 384, 384).cuda()
    net = RetinexNet(stage1_path="../ckpt/stage1_weight.pth").cuda().eval()

    with torch.no_grad():
        # print params and flops
        flops, params = profile(net, inputs=(x1, x2))
        print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
        print("params=", str(params / 1e6) + '{}'.format("M"))

        # total = sum(p.numel() for p in net.parameters())
        # print("Total params: %.2fM" % (total / 1e6))

        out = net(x1, x2)
        print(out[0].shape)

