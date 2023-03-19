# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:32:47
# @FileName:  real_model.py
# @Contact :  lianghao@whu.edu.cn

import os
import torch
import cv2 as cv
from skimage import img_as_ubyte
from util.log_util import Recorder
import util.data_util as data_util
from einops.layers.torch import Rearrange
from util.metric_util import standard_psnr_ssim
from net.select_net import Net
import util.data_util as data_util
from util.train_util import resume_state
from dataset.basic_dataset.basic_unpair import BasicDataSetUnPair

class BasicModel:
    def __init__(self, option, logger, main_dir):
        self.option = option
        self.logger = logger
        self.main_dir = main_dir
        self.recoder = Recorder(option)
        self.save = option['test']['save']
        self.gpu = option['test']['gpu']
        # self.mode = option['test']['mode']
        self.save_dir = os.path.join(main_dir, option['directory']['vision'])
        self.net = Net(option=option)()
        self.dataset_test = RealDataSet(input_dir=option['dataset']['test']['input'],
                                        mode=option['network']['mode'])

        self.__resume__()
        if self.gpu:
            self.net = self.net.cuda()
        logger.info("Every Thing has been prepared . ")

    def __resume__(self):
        mode = self.option['global_setting']['resume']['mode']
        checkpoint = self.option['global_setting']['resume']['checkpoint']
        self.net = resume_state(checkpoint, net=self.net, mode=mode)

    def __save_tensor__(self, name, tensor):
        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(0, 1))
        save_path = os.path.join(self.save_dir, name)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)

    def test(self, name, data_pair):
        self.net.eval()
        with torch.no_grad():
            input = data_pair
            input = Rearrange('c h w -> (1) c h w')(input)
            if self.gpu:
                input = input.cuda()
            out = self.net(input)

        if self.save:
            self.__save_tensor__(name, tensor=out)


class RealDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir, mode='gray'):
        super(RealDataSet, self).__init__(input_dir, mode)

    def __getitem__(self, item):
        input = self.__get_image__(item)
        tensor_in = data_util.image2tensor(input)
        return tensor_in