# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:30:13
# @FileName:  pair_real.py
# @Contact :  lianghao@whu.edu.cn

import util.data_util as data_util
from dataset.basic_dataset.basic_pair import BasicDataSetPair


class TrainDataSet(BasicDataSetPair):
    def __init__(self, input_dir, target_dir, patch_size, mode='gray'):
        super(TrainDataSet, self).__init__(input_dir, target_dir, mode=mode)
        self.patch_size = patch_size

    def __getitem__(self, item):
        input, target = self.__get_image__(item)
        input, target = data_util.random_augmentation(input, target)
        patch_in, patch_tar = data_util.random_image2patch(input, target, patch_size=self.patch_size)
        tensor_in = data_util.image2tensor(patch_in)
        tensor_tar = data_util.image2tensor(patch_tar)
        return tensor_in, tensor_tar

class TestDataSet(BasicDataSetPair):
    def __init__(self, input_dir, target_dir, mode='gray'):
        super(TestDataSet, self).__init__(input_dir, target_dir, mode=mode)

    def __getitem__(self, item):
        input, target = self.__get_image__(item)
        tensor_in = data_util.image2tensor(input)
        tensor_tar = data_util.image2tensor(target)
        return tensor_in, tensor_tar