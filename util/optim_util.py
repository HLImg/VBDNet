# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:18:03
# @FileName:  optim_util.py
# @Contact :  lianghao@whu.edu.cn

import torch.optim as optim


class Optimizer:
    def __init__(self, model, option):
        self.name = option['train']['optim']['name']
        self.params = option['train']['optim']['params']
        self.params['params'] = model.parameters()

    def __call__(self):
        if self.name == 'SGD':
            return optim.SGD(**self.params)
        elif self.name == 'Adam':
            return optim.Adam(**self.params)
        else:
            assert 1 == 2, f"the name of optimizer <{self.name}> is incorrect"


class Scheduler:
    def __init__(self, optimizer, option):
        self.name = option['train']['scheduler']['name']
        self.params = option['train']['scheduler']['params']
        self.params['optimizer'] = optimizer

    def __call__(self):
        if self.name == 'ExponentialLR':
            return optim.lr_scheduler.ExponentialLR(**self.params)
        elif self.name == 'StepLR':
            return optim.lr_scheduler.StepLR(**self.params)
        elif self.name == 'MultiStepLR':
            return optim.lr_scheduler.MultiStepLR(**self.params)
        elif self.name == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(**self.params)
        else:
            assert 1 == 2, f"the name of scheduler <{self.name}> is incorrect"
