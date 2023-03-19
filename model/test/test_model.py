# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:33:08
# @FileName:  test_model.py
# @Contact :  lianghao@whu.edu.cn

import os
import torch
import cv2 as cv
from net.select_net import Net
from skimage import img_as_ubyte
from util.log_util import Recorder
import util.data_util as data_util
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from dataset.select_dataset import DataSet
from util.metric_util import standard_psnr_ssim
from loss.image_loss import PerceptralLoss
from util.train_util import resume_state


class BasicModel:
    def __init__(self, option, logger, main_dir):
        self.option = option
        self.logger = logger
        self.main_dir = main_dir
        self.recoder = Recorder(option)
        self.save = option['test']['save']
        self.gpu = option['test']['gpu']
        # self.mode = option['network']['mode']
        self.mode = option['test']['metric_mode']
        self.save_dir = os.path.join(main_dir, option['directory']['vision'])
        self.net = Net(option=option)()
        _, self.dataset_test = DataSet(option)()
        self.loader_test = DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
        self.percepLoss = PerceptralLoss()

        self.__resume__()
        if self.gpu:
            self.net = self.net.cuda()
            self.percepLoss = self.percepLoss.cuda()

        logger.info("Every Thing has been prepared . ")

    def __resume__(self):
        mode = self.option['global_setting']['resume']['mode']
        checkpoint = self.option['global_setting']['resume']['checkpoint']
        self.net = resume_state(checkpoint, net=self.net, mode=mode)

    def __save_tensor__(self, name, tensor):
        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        save_path = os.path.join(self.save_dir, name)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)

    def test(self, name, data_pair):
        self.net.eval()
        with torch.no_grad():
            input, target = [x for x in data_pair]
            input = Rearrange('c h w -> (1) c h w')(input)
            target = Rearrange('c h w -> (1) c h w')(target)
            if self.gpu:
                input, target = input.cuda(), target.cuda()
            output = self.net(input)

        if self.save:
            self.__save_tensor__(name, tensor=output)
        psnr, ssim = standard_psnr_ssim(input=output, target=target, mode=self.mode)
        with torch.no_grad():
            if output.size(1) == 1:
                output = torch.cat([output, output, output], dim=1)
                target = torch.cat([target, target, target], dim=1)
            percep_loss = self.percepLoss(output, target)
        return psnr, ssim, percep_loss.item()