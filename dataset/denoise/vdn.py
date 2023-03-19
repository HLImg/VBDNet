# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:31:30
# @FileName:  vdn.py
# @Contact :  lianghao@whu.edu.cn

import cv2  as cv
import numpy as np
import torch

import util.data_util as data_util
from dataset.basic_dataset.basic_unpair import BasicDataSetUnPair

def sigma_estimate(im_noisy, im_gt, win, sigma_spatial):
    noise2 = (im_noisy - im_gt) ** 2
    sigma2_map_est = cv.GaussianBlur(noise2, (win, win), sigma_spatial)
    sigma2_map_est = sigma2_map_est.astype(np.float32)
    sigma2_map_est = np.where(sigma2_map_est<1e-10, 1e-10, sigma2_map_est)
    if sigma2_map_est.ndim == 2:
        sigma2_map_est = sigma2_map_est[:, :, np.newaxis]
    return sigma2_map_est

class TrainDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir, patch_size, levels, task='poisson', mode='gray', clip=False):
        super(TrainDataSet, self).__init__(input_dir=input_dir, mode=mode)
        self.clip = clip
        self.levels = levels
        self.patch_size = patch_size
        if task.upper() == 'poisson'.upper():
            self.add_noise = data_util.add_poisson_noise
        elif task.upper() == 'gaussian'.upper():
            self.add_noise = data_util.add_gaussian_noise
        self.eps2 = 1e-6
        self.sigma_spatial = 3
        self.window_size = 2 * self.sigma_spatial + 1

    def __getitem__(self, item):
        level = np.random.choice(self.levels)
        gt = self.__get_image__(item)
        gt, = data_util.random_augmentation(gt)
        patch_g, = data_util.random_image2patch(gt, patch_size=self.patch_size)
        patch_n = self.add_noise(patch_g.copy(), level=level, clip=self.clip)
        sigma2_map_est = sigma_estimate(im_noisy=patch_n, im_gt=patch_g, win=self.window_size,
                                        sigma_spatial=self.sigma_spatial)
        tensor_g = data_util.image2tensor(patch_g)
        tensor_n = data_util.image2tensor(patch_n)
        tensor_map = data_util.image2tensor(sigma2_map_est)
        tensor_eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1, 1, ))
        return tensor_n, tensor_g, tensor_map, tensor_eps2

class TestDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir, levels, task='poisson', mode='gray', clip=False):
        super(TestDataSet, self).__init__(input_dir, mode=mode)
        self.clip = clip
        self.levels = levels
        if task.upper() == 'poisson'.upper():
            self.add_noise = data_util.add_poisson_noise
        elif task.upper() == 'gaussian'.upper():
            self.add_noise = data_util.add_gaussian_noise

    def __getitem__(self, item):
        level = np.random.choice(self.levels)
        gt = self.__get_image__(item)
        noisy = self.add_noise(gt.copy(), level=level, clip=self.clip)
        tensor_g = data_util.image2tensor(gt)
        tensor_n = data_util.image2tensor(noisy)
        return tensor_n, tensor_g