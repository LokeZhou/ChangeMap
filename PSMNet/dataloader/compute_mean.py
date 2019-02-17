# coding: utf-8

import numpy as np
import cv2
import random
from listflowfile import *


"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


#train_txt_path = '../predata/train.txt'

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataloader("../predata")

#CNum = 2000     # 挑选多少图片进行计算



img_h, img_w = 512, 512
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []


for i in range(len(all_left_img)):
        img_path = all_left_img[i]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        print(i)

for i in range(len(all_right_img)):
        img_path = all_right_img[i]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        print(i)


imgs = imgs.astype(np.float32)/255.


for i in range(3):
    pixels = imgs[:,:,i,:].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))