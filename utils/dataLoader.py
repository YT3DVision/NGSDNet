import os
import os.path
import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import random
import numpy as np
import cv2


def RandomCrop(rgb, nir, gt, illu, illu2):
    border = 30
    image_width    = rgb.size[0]
    image_height   = rgb.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return rgb.crop(random_region), nir.crop(random_region), gt.crop(random_region), illu.crop(random_region), illu2.crop(random_region)


def RandomHorizontalFlip(rgb, nir, gt, illu, illu2):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        nir = nir.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        illu = illu.transpose(Image.FLIP_LEFT_RIGHT)
        illu2 = illu2.transpose(Image.FLIP_LEFT_RIGHT)
    return rgb, nir, gt, illu, illu2


def RandomRotation(rgb, nir, gt, illu, illu2):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        rgb = rgb.rotate(random_angle, mode)
        nir = nir.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
        illu = illu.rotate(random_angle, mode)
        illu2 = illu2.rotate(random_angle, mode)
    return rgb, nir, gt, illu, illu2


def make_train_data(data_path):
    print("INFO: Processing Train Data")
    # data_path = os.path.join(data_path, 'train')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "train_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "train_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
            os.path.join(data_path, "i", img_name + ".png"),
            os.path.join(data_path, "r", img_name + ".png"),
            os.path.join(data_path, "illu2", img_name + ".png"),
        )
        for img_name in img_list
    ]


def make_test_data(data_path):
    print("INFO: Processing Test Data")
    # data_path = os.path.join(data_path, 'test')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "test_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "test_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
            os.path.join(data_path, "i", img_name + ".png"),
            os.path.join(data_path, "r", img_name + ".png"),
            os.path.join(data_path, "illu2", img_name + ".png"),
        )
        for img_name in img_list
    ]

def make_train_data2(data_path):
    print("INFO: Processing Train Data")
    # data_path = os.path.join(data_path, 'train')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "train_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "train_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
            os.path.join(data_path, "edge", img_name + ".png"),
        )
        for img_name in img_list
    ]


def make_test_data2(data_path):
    print("INFO: Processing Test Data")
    # data_path = os.path.join(data_path, 'test')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "test_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "test_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
            os.path.join(data_path, "edge", img_name + ".png"),
        )
        for img_name in img_list
    ]


def make_trans10k_data(data_path, train):
    print("INFO: Processing Test Data")
    if train:
        data_path = os.path.join(data_path, 'train')
    else:
        data_path = os.path.join(data_path, 'test')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "images"))
        if f.endswith(".jpg")
    ]
    return [
        (
            os.path.join(data_path, "images", img_name + ".jpg"),
            os.path.join(data_path, "masks", img_name + "_mask.png"),
            os.path.join(data_path, "images", img_name + ".jpg"),
            os.path.join(data_path, "edges", img_name + "_mask.png"),
        )
        for img_name in img_list
    ]


class make_dataSet(data.Dataset):
    def __init__(self, data_path, train=True, rgb_transform=None, grey_transform=None):
        self.train = train
        self.data_path = data_path
        self.rgb_transform = rgb_transform
        self.grey_transform = grey_transform
        self.images = make_trans10k_data(data_path, train)
        # if self.train:
        #     self.images = make_train_data2(data_path)
        # else:
        #     self.images = make_test_data2(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image_path, glass_path, nir_path, i_path, r_path, illu2_path = self.images[index]
        image_path, glass_path, nir_path, edge_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        glass = Image.open(glass_path).convert("L")
        nir = Image.open(nir_path).convert("RGB")
        edge = Image.open(edge_path).convert("L")
        # illu = Image.open(i_path).convert("L")
        # reflect = Image.open(r_path).convert("RGB")
        # illu2 = Image.open(illu2_path).convert("L")

        glass = np.array(glass)
        glass[glass != 0] = 255
        glass = Image.fromarray(glass)

        edge = np.array(edge)
        edge[edge != 0] = 255
        edge = Image.fromarray(edge)
        # data augumentation
        # if self.train:
        #     reflect, nir, glass, illu, illu2 = RandomCrop(reflect, nir, glass, illu, illu2)
        #     reflect, nir, glass, illu, illu2 = RandomHorizontalFlip(reflect, nir, glass, illu, illu2)
        #     reflect, nir, glass, illu, illu2 = RandomRotation(reflect, nir, glass, illu, illu2)

        if self.rgb_transform is not None:
            image = self.rgb_transform(image)
            nir = self.rgb_transform(nir)
            # reflect = self.rgb_transform(reflect)

        if self.grey_transform is not None:
            glass = self.grey_transform(glass)
            edge = self.grey_transform(edge)
            # illu = self.grey_transform(illu)
            # illu2 = self.grey_transform(illu2)

        # if self.train:
        #     return image, glass, nir, illu, illu2, reflect
        # else:
        #     return image, glass, nir, illu, illu2, reflect
        if self.train:
            return image, glass, nir, edge
        else:
            return image, glass, nir, edge