# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:17:40
# @FileName:  metric_util.py
# @Contact :  lianghao@whu.edu.cn

import cv2 as cv
import numpy as np
import torch.nn as nn
from skimage import img_as_ubyte
from torchvision.models import vgg19
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def standard_psnr_ssim(input, target, mode='y'):
    """
    psnr and ssim
    @param input: tensor (1, c, h, w)
    @param target: tensor (1, c, h, w)
    @param mode: ['gray', 'rgb', 'ycbcy']
    @return: psnr value and ssim value
    """
    input = input.data.cpu().numpy().clip(0, 1)
    target = target.data.cpu().numpy().clip(0, 1)
    input = np.transpose(input[0, :], (1, 2, 0))
    target = np.transpose(target[0, :], (1, 2, 0))
    input, target = img_as_ubyte(input), img_as_ubyte(target)
    if mode == 'y':
        input = cv.cvtColor(input, cv.COLOR_RGB2YCrCb)[:, :, :1]
        target = cv.cvtColor(target, cv.COLOR_RGB2YCrCb)[:, :, :1]

    psnr = peak_signal_noise_ratio(image_true=target, image_test=input)
    ssim = structural_similarity(im1=input, im2=target, channel_axis=2)
    return psnr, ssim