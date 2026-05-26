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

from model.MyNetv15B import NIRNet
from utils.dataLoader import *
from utils.Miou import iou_mean
from utils.loss import *
import warnings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    # set random seed = 42
    seed = 42
    set_random_seed(seed)

    train_min_loss = float("inf")
    valid_max_glass = float(0)

    # 输入图像的尺寸大小，默认384
    scale = 384

    parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
    parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=300, help="")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    parser.add_argument("--data_path", type=str, default="E:/dataset/trans10k", help="")
    parser.add_argument("--save_path", type=str, default="./ckpt_10k/", help="")
    parser.add_argument("--lr", type=float, default=2e-5, help="")

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_obj = torch.cuda.get_device_properties(device)
        print(f"Using CUDA device {device}: {device_obj.name}")
    else:
        print("CUDA is not available. Using CPU.")

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

    model = NIRNet(
        backbone_path="./model/backbone/swin_transformer/swinv2_base_patch4_window24_384.pth",
        stage1_path="./ckpt/stage1_weight.pth",
    ).cuda()
    model.load_state_dict(torch.load("./ckpt/best.pth"), strict=False)
    print("model initiating success")

    for name, param in model.named_parameters():
        if "decom" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # print params and flops
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    # initiate optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(opt.epochs):
        start = time.time()
        model.train()
        model.zero_grad()

        train_loss_sum = 0
        t_glass_iou = 0
        iters = 0

        for idx, (rgb, glass, nir, edge) in enumerate(loader_train, 0):
            iters += 1
            rgb = Variable(rgb).cuda()
            nir = Variable(nir).cuda()
            glass = Variable(glass).cuda()
            edge = Variable(edge).cuda()

            optimizer.zero_grad()
            g3, g2, g1, g0, g_fuse, g_final, auxi0, auxi1, edge0, edge1, edge2, edge3 = model(rgb, nir)

            # Calculate Loss
            # predict loss
            loss3 = bce_iou_loss(g3, glass)
            loss2 = bce_iou_loss(g2, glass)
            loss1 = bce_iou_loss(g1, glass)
            loss0 = bce_iou_loss(g0, glass)
            lossfuse = bce_iou_loss(g_fuse, glass)
            lossfinal = bce_iou_loss(g_final, glass)
            predict_loss = lossfinal +1 * loss3 + \
                           2 * loss2 + \
                           3 * loss1 + \
                           4 * loss0 + \
                           5 * lossfinal + \
                           10 * lossfuse

            # auxiliary loss
            loss_auxi0 = bce_iou_loss(auxi0, glass)
            loss_auxi1 = dice_loss(auxi1, glass)
            loss_edge0 = dice_loss(edge0, edge)
            loss_edge1 = dice_loss(edge1, edge)
            loss_edge2 = dice_loss(edge2, edge)
            loss_edge3 = dice_loss(edge3, edge)
            auxi_loss = 1 * loss_auxi0 + \
                        1 * loss_auxi1 + \
                        1 * loss_edge0 + \
                        1 * loss_edge1 + \
                        1 * loss_edge2 + \
                        1 * loss_edge3
                        
            # Loss Backward
            loss = predict_loss + 0.1 * auxi_loss
            loss.backward()
            train_loss_sum = loss

            optimizer.step()
            scheduler.step()

            if iters % (500//opt.batchSize) == 0:
                print('epoch: [%2d/%2d], iter: [%3d/%3d] || loss : %5.6f || time : %f' % (
                    epoch+1, opt.epochs, iters, int(len(dataset_train))//opt.batchSize, train_loss_sum / 100, time.time()-start))

        model.eval()
        model.zero_grad()

        v_glass_iou = 0
        valid_loss_sum = 0

        with (torch.no_grad()):
            for idx, (rgb, glass, nir, edge) in enumerate(loader_valid, 0):
                rgb = Variable(rgb).cuda()
                nir = Variable(nir).cuda()
                glass = Variable(glass).cuda()

                g_final = model(rgb, nir)

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
            "INFO: epoch:{},train loss:{},validation iou:{}, time:{}".format(
                epoch + 1, train_loss_sum / len(loader_train), v_glass_iou / len(loader_valid) / opt.batchSize * 100,
                round(t, 2),
            )
        )

        if v_glass_iou > valid_max_glass:
            valid_max_glass = v_glass_iou
            torch.save(model.state_dict(), os.path.join(opt.save_path, "val_iou_max.pth"))
            print("INFO: save validation_iou_max model")

        torch.save(model.state_dict(), os.path.join(opt.save_path, "latest.pth"))
