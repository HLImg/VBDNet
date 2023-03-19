# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:16:56
# @FileName:  data_util.py
# @Contact :  lianghao@whu.edu.cn

import os
import glob
import torch
import random
import cv2 as cv
import numpy as np
from skimage import img_as_float32, img_as_ubyte


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def image2tensor(image):
    assert image is not None, 'the image is none'
    if len(image.shape) == 2:
        image = image[np.newaxis, :]
    else:
        image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image.copy())


def tensor2image(tensor):
    assert len(tensor.shape) == 3 or len(tensor.shape) == 4, f'tensor shape is {tensor.shape}'
    tensor = tensor.data.cpu().numpy()
    if len(tensor.shape) == 4:
        image = np.transpose(tensor, (0, 2, 3, 1))
    else:
        image = np.transpose(tensor, (1, 2, 0))
    return image


def get_image_path(directory):
    assert os.path.exists(directory), f'The {directory} is not exist'
    paths = [path for path in glob.glob(os.path.join(directory, '*'))]
    paths = np.array(sorted(paths))
    return paths


def augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.fliplr(image)
    elif mode == 2:
        out = np.flipud(image)
    elif mode == 3:
        out = np.fliplr(image)
        out = np.rot90(out, k=1)
    elif mode == 4:
        out = np.flipud(image)
        out = np.rot90(out, k=2)
    elif mode == 5:
        out = np.fliplr(image)
        out = np.rot90(out, k=3)
    elif mode == 6:
        out = np.flipud(image)
        out = np.rot90(out, k=3)
    return out


def random_augmentation(*images):
    mode = random.randint(0, 6)
    out = []
    for image in images:
        out.append(augmentation(image, mode))
    return out


def random_image2patch(*images, patch_size):
    """
    random crop image to patch with patch-size
    :param images: [(h, w, c)] or [(h, w)]
    :param patch_size: 128
    :return: [(patch-size, patch-size, c)] or [(patch_size, patch-size)]
    """
    h, w = images[0].shape[:2]
    max_h = max(h, patch_size)
    max_w = max(w, patch_size)
    ind_h = random.randint(0, max_h - patch_size)
    ind_w = random.randint(0, max_w - patch_size)
    out = []
    for image in images:
        if len(image.shape) == 2:
            patch = image[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
        else:
            patch = image[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
        out.append(patch)
    return out


def random_image2patch_sr(lr_img, hr_img, patch_size, up_scale=2):
    lr_h, lr_w = lr_img.shape[:2]
    hr_h, hr_w = hr_img.shape[:2]
    h = min(lr_h, hr_h // up_scale)
    w = min(lr_w, hr_w // up_scale)
    max_h = max(h, patch_size)
    max_w = max(w, patch_size)
    ind_h = random.randint(0, max_h - patch_size)
    ind_w = random.randint(0, max_w - patch_size)
    if len(lr_img.shape) == 2:
        lr_patch = lr_img[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
        hr_patch = hr_img[ind_h * up_scale: (ind_h +
                                             patch_size) * up_scale, ind_w * up_scale: (ind_w + patch_size) * up_scale]
    else:
        lr_patch = lr_img[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size, :]
        hr_patch = hr_img[ind_h * up_scale: (ind_h + patch_size) * up_scale, ind_w * up_scale: (ind_w + patch_size) * up_scale, :]
    return lr_patch, hr_patch


def padding(x, bench_size=8, position='mid', mode='replicated', value=0):
    h, w = x.size()[-2:]
    ch, cw = h // bench_size, w // bench_size
    if ch * bench_size < h:
        ch = ch + 1
    if cw * bench_size < w:
        cw = cw + 1
    ch = ch * bench_size
    cw = cw * bench_size
    if position.upper() == 'mid'.upper():
        padding_bottom = (ch - h) // 2
        padding_top = ch - h - (ch - h) // 2
        padding_right = (cw - w) // 2
        padding_left = cw - w - (cw - w) // 2
    elif position.upper() == 'left-up'.upper():
        padding_bottom = ch - h
        padding_top = 0
        padding_right = cw - w
        padding_left = 0


    padding = (padding_left, padding_right, padding_top, padding_bottom)
    if mode.upper() == 'constant'.upper():
        x = torch.nn.ConstantPad2d(padding=padding, value=value)(x)
    elif mode.upper() == 'replicated'.upper():
        x = torch.nn.ReplicationPad2d(padding=padding)(x)
    elif mode.upper() == 'reflected'.upper():
        x = torch.nn.ReflectionPad2d(padding=padding)(x)
    elif mode.upper() == 'zero'.upper():
        x = torch.nn.ZeroPad2d(padding)(x)

    return h, w, padding_top, padding_left, x


def add_gaussian_noise(image, level, clip=False):
    noise = np.random.normal(0, level / 255., image.shape)
    noisy = np.float32(image + noise)
    if clip:
        noisy = noisy.clip(0, 1)
    return noisy


def add_poisson_noise(image, level, clip=False):
    noisy = np.float32(np.random.poisson(image * level) / level)
    if clip:
        noisy = noisy.clip(0, 1)
    return noisy

