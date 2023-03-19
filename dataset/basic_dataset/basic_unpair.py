# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:29:39
# @FileName:  basic_unpair.py
# @Contact :  lianghao@whu.edu.cn

import cv2 as cv
import util.data_util as data_util
from torch.utils.data import Dataset
from skimage import img_as_float32


class BasicDataSetUnPair(Dataset):
    def __init__(self, input_dir, mode='gray'):
        super(BasicDataSetUnPair, self).__init__()
        self.mode = mode
        self.input_paths = data_util.get_image_path(directory=input_dir)

    def __len__(self):
        return len(self.input_paths)

    def __get_image__(self, item):
        input = cv.imread(self.input_paths[item], flags=-1)
        if input is None:
            print(self.input_paths[item])
            re = cv.imread(self.input_paths[item], flags=-1)
            input = re
        assert input is not None, f'the image from {self.input_paths[item]} is empty'

        if len(input.shape) == 3:
            if self.mode == 'gray':
                input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
            elif self.mode == 'rgb':
                input = cv.cvtColor(input, cv.COLOR_BGR2RGB)

        return img_as_float32(input)