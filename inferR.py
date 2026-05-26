import argparse
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
import os
from model.RetinexNet import RetinexNet
from utils.Miou import *

# 输入图像的尺寸大小，默认384
scale = 384

parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--data_path", type=str, default="E:/dataset/Nighttime/sample", help="")
parser.add_argument("--save_path", type=str, default="./ckpt", help="")
parser.add_argument("--result_path", type=str, default="E:/dataset/Nighttime/sample", help="")


opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

img_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

target_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

to_pil = transforms.ToPILImage()

if not os.path.isdir(opt.result_path):
    os.makedirs(opt.result_path)


def main():
    # ######## create Model #############
    model = RetinexNet(stage1_path="./ckpt/stage1_weight.pth").cuda()

    # print params
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    model.eval()
    with torch.no_grad():

        start = time.time()
        # img_list = [
        #     img_name for img_name in os.listdir(os.path.join(opt.data_path, "edge"))
        # ]
        img_list = os.listdir(opt.data_path)
        print("test image pairs:%d" % len(img_list))

        for idx, img_name in enumerate(img_list):
            nir = Image.open(os.path.join(opt.data_path, img_name, 'nir.png')).convert("RGB")
            rgb = Image.open(os.path.join(opt.data_path, img_name, 'rgb.png')).convert("RGB")

            nir = Variable(img_transform(nir).unsqueeze(0)).cuda()
            rgb = Variable(img_transform(rgb).unsqueeze(0)).cuda()
            start_time = time.time()
            refl, illu, refl2, illu2 = model(rgb, nir)
            end_time = time.time()
            print(end_time - start_time)
            refl = refl.data.squeeze(0)
            refl2 = refl2.data.squeeze(0)
            illu = illu.data.squeeze(0)
            illu2 = illu2.data.squeeze(0)

            refl = np.array(transforms.Resize((scale, scale))(to_pil(refl)))
            refl2 = np.array(transforms.Resize((scale, scale))(to_pil(refl2)))
            illu = np.array(transforms.Resize((scale, scale))(to_pil(illu)))
            illu2 = np.array(transforms.Resize((scale, scale))(to_pil(illu2)))

            Image.fromarray(refl).convert("RGB").save(os.path.join(opt.result_path, img_name, "r.png"))
            Image.fromarray(refl2).convert("L").save(os.path.join(opt.result_path, img_name, "r2.png"))
            Image.fromarray(illu).convert("L").save(os.path.join(opt.result_path, img_name, "i.png"))
            Image.fromarray(illu2).convert("L").save(os.path.join(opt.result_path, img_name, "i2.png"))
            print(img_name[:-4])

        end = time.time()
        print("Average Time Is : {:.6f}".format((end - start) / len(img_list)))

main()
