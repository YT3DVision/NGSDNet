import argparse
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
import os
from model.MyNet import NIRNet
from utils.Miou import *

# 输入图像的尺寸大小，默认384
scale = 256

#
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--data_path", type=str, default="./data", help="")
parser.add_argument("--save_path", type=str, default="./ckpt", help="")
parser.add_argument("--result_path", type=str, default="./results", help="")


opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

img_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

target_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

to_pil = transforms.ToPILImage()

# glass_path = os.path.join(opt.result_path, "ghost")

if not os.path.isdir(opt.result_path):
    os.makedirs(opt.result_path)


def main():
    # ######## create Model #############
    model = NIRNet().cuda()

    model.load_state_dict(torch.load(os.path.join(opt.save_path, "finetune_iou_max.pth")), strict=False)

    # print params
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    model.eval()
    with torch.no_grad():

        start = time.time()
        img_list = [
            img_name for img_name in os.listdir(os.path.join(opt.data_path, "test_gt"))
        ]
        print("test image pairs:%d" % len(img_list))

        v_glass_iou = 0

        for idx, img_name in enumerate(img_list):
            img = Image.open(os.path.join(opt.data_path, "r", img_name))
            nir = Image.open(os.path.join(opt.data_path, "nir", img_name)).convert("RGB")
            glass = Image.open(os.path.join(opt.data_path, "test_gt", img_name)).convert("L")

            rgb = Variable(img_transform(img).unsqueeze(0)).cuda()
            nir = Variable(img_transform(nir).unsqueeze(0)).cuda()
            glass = Variable(img_transform(glass).unsqueeze(0)).cuda()

            g3, g2, g1, g0, g_fuse, g_final = model(rgb, nir)

            pred = torch.round(g_final).squeeze(0).int().detach().cpu()
            target = torch.round(glass).squeeze(0).int().detach().cpu()

            v_glass_iou += iou_mean(pred, target, 1)

            g_final = g_final.data.squeeze(0)
            g_final = np.array(transforms.Resize((scale, scale))(to_pil(g_final)))

            Image.fromarray(g_final).save(
                os.path.join(opt.result_path, img_name[:-4] + ".png")
            )

        end = time.time()
        print("Average Time Is : {:.6f}".format((end - start) / len(img_list)))
        print("Test IoU is : {:.2f}".format(v_glass_iou / len(img_list) * 100))


main()
