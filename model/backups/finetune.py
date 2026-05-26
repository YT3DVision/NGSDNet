import argparse
import torch.optim as optim
import torch
import time
import os

from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from thop import profile
from model.MyNet import NIRNet
from utils.dataLoader import *
from utils.Miou import iou_mean
from utils.loss import *
import warnings


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_random_seed(42)

    train_min_loss = float("inf")
    valid_max_glass = float(0)

    # 输入图像的尺寸大小，默认384
    scale = 256

    parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
    parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=300, help="")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    parser.add_argument("--data_path", type=str, default="./data", help="")
    parser.add_argument("--save_path", type=str, default="./ckpt/", help="")
    parser.add_argument("--finetune_model_path", type=str, default="./ckpt/epoch30.pth", help="")
    parser.add_argument("--lr", type=float, default=1e-5, help="")

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    rgb_transform = transforms.Compose(
        [transforms.Resize((scale, scale)), transforms.ToTensor()]
    )

    grey_transform = transforms.Compose(
        [transforms.Resize((scale, scale)), transforms.ToTensor()]
    )

    # load dataset
    print("INFO:Loading dataset ...\n")
    dataset_train = make_dataSet(
        opt.data_path,
        train=True,
        rgb_transform=rgb_transform,
        grey_transform=grey_transform,
    )
    dataset_valid = make_dataSet(
        opt.data_path,
        train=False,
        rgb_transform=rgb_transform,
        grey_transform=grey_transform,
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        num_workers=1,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
    )
    loader_valid = DataLoader(
        dataset=dataset_valid, num_workers=1, batch_size=opt.batchSize, shuffle=False
    )
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of valid samples: %d\n" % int(len(dataset_valid)))

    # initiate Model
    model = NIRNet().cuda()
    model.load_state_dict(torch.load(opt.finetune_model_path), strict=False)
    print("model loading success")

    # print params
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    # initiate optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        start = time.time()
        model.train()
        model.zero_grad()

        train_loss_sum = 0
        t_glass_iou = 0
        iters = 0

        for idx, (input_data, glass, nir) in enumerate(loader_train, 0):
            iters += 1

            input_data = Variable(input_data)
            nir = Variable(nir)
            glass = Variable(glass)

            input_data = input_data.cuda()
            nir = nir.cuda()
            glass = glass.cuda()

            optimizer.zero_grad()
            g3, g2, g1, g0, g_fuse, g_final = model(input_data, nir)

            loss3 = bce_iou_loss(g3, glass)
            loss2 = bce_iou_loss(g2, glass)
            loss1 = bce_iou_loss(g1, glass)
            loss0 = bce_iou_loss(g0, glass)
            lossfuse = bce_iou_loss(g_fuse, glass)
            lossfinal = bce_iou_loss(g_final, glass)

            loss = loss3 + 2*loss2 + 3*loss1 + 4*loss0 + 5*lossfinal + lossfuse
            loss.backward()
            train_loss_sum = loss

            optimizer.step()

            if iters % (300//opt.batchSize) == 0:
                print('epoch: [%2d/%2d], iter: [%3d/%3d] || loss : %5.6f || time : %f' % (
                    epoch+1, opt.epochs, iters, int(len(dataset_train))//opt.batchSize, train_loss_sum / 100, time.time()-start))

        model.eval()
        model.zero_grad()

        v_glass_iou = 0
        valid_loss_sum = 0

        with torch.no_grad():
            for idx, (input_data, glass, nir) in enumerate(loader_valid, 0):
                input_data = Variable(input_data)
                nir = Variable(nir)
                glass = Variable(glass)

                input_data = input_data.cuda()
                nir = nir.cuda()
                glass = glass.cuda()

                g3, g2, g1, g0, g_fuse, g_final = model(input_data, nir)

                pred = g_final
                label = glass
                bs, _, _, _ = label.shape

                temp1 = pred.data.squeeze(1)
                temp2 = label.data.squeeze(1)
                for i in range(bs):
                    a = temp1[i, :, :]
                    b = temp2[i, :, :]
                    a = torch.round(a).squeeze(0).int().detach().cpu()
                    b = torch.round(b).squeeze(0).int().detach().cpu()
                    v_glass_iou += iou_mean(a, b, 1)

                torch.cuda.empty_cache()

        end = time.time()
        t = end - start

        print(
            "INFO: epoch:{},finetune loss:{},validation iou:{}, time:{}".format(
                epoch + 1, train_loss_sum / len(loader_train), v_glass_iou / len(loader_valid) / opt.batchSize * 100,
                round(t, 2),
            )
        )

        if train_loss_sum < train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(model.state_dict(), os.path.join(opt.save_path, "finetune_loss_min.pth"))
            print("INFO: save finetune_loss_min model")

        if v_glass_iou > valid_max_glass:
            valid_max_glass = v_glass_iou
            torch.save(model.state_dict(), os.path.join(opt.save_path, "finetune_iou_max.pth"))
            print("INFO: save finetune_test_iou_max model")



