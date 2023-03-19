# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:30:58
# @FileName:  denoise_unpair.py
# @Contact :  lianghao@whu.edu.cn

import numpy as np
import util.data_util as data_util
from dataset.basic_dataset.basic_unpair import BasicDataSetUnPair


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

    def __getitem__(self, item):
        level = np.random.choice(self.levels)
        gt = self.__get_image__(item)
        gt, = data_util.random_augmentation(gt)
        patch_g, = data_util.random_image2patch(gt, patch_size=self.patch_size)
        patch_n = self.add_noise(patch_g.copy(), level=level, clip=self.clip)
        tensor_g = data_util.image2tensor(patch_g)
        tensor_n = data_util.image2tensor(patch_n)
        return tensor_n, tensor_g


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
