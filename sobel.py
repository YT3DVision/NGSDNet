import cv2
import numpy as np
import os

input_dir = 'E:/dataset/trans_10k_things/train/mask'
output_dir = 'E:/dataset/trans_10k_things/train/edge'
def compute_edge(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        # save_path = os.path.join(output_dir, img_name)
        yuan = cv2.imread(img_path)
        # yuan = cv2.resize(yuan, (518, 518))
        # 使用Sobel算子进行水平边缘检测
        yuan_x_64 = cv2.Sobel(yuan, cv2.CV_64F, dx=1, dy=0)
        # 这一步将默认的uint8数据类型更改为float64，以便能够保存负数的边缘强度
        # 将Sobel水平边缘检测结果的负数值转换为正数
        yuan_x_full = cv2.convertScaleAbs(yuan_x_64)
        # 这一步将负数值转换为其绝对值，以便在显示时产生正确的视觉效果
        # 使用Sobel算子进行垂直边缘检测
        yuan_y_64 = cv2.Sobel(yuan, cv2.CV_64F, dx=0, dy=1)
        # 这一步将默认的uint8数据类型更改为float64，以便能够保存负数的边缘强度
        # 将Sobel垂直边缘检测结果的负数值转换为正数
        yuan_y_full = cv2.convertScaleAbs(yuan_y_64)
        # 这一步将负数值转换为其绝对值，以便在显示时产生正确的视觉效果
        # 使用addWeighted函数将水平和垂直边缘检测结果叠加，创建合并的边缘检测图像
        yuan_xy_full = cv2.addWeighted(yuan_x_full, 1, yuan_y_full, 1, 0)

        # 膨胀操作增粗边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dst = cv2.dilate(yuan_xy_full, kernel, iterations=1)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, dst)
        print(img_name.split('.')[0] + ' is sobel-filtered.')

video_path = "E:/dataset/video_glass_reflection_motion_dataset_0327/train"
for video_name in os.listdir(video_path):
    mask_path = os.path.join(video_path, video_name, "SegmentationClassPNG")
    edge_path = os.path.join(video_path, video_name, "edge")
    os.makedirs(edge_path, exist_ok=True)
    compute_edge(mask_path, edge_path)

print('finished')
