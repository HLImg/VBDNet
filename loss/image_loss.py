# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:22 PM
# @File    : image_loss.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss to approximate l1-norm"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = torch.add(target, -input)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(input=img, pad=(kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(input=img, weight=self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        # downsample
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, input, target):
        loss = self.loss(self.laplacian_kernel(target), self.laplacian_kernel(input))
        return loss

class PerceptralLoss(nn.Module):
    def __init__(self):
        super(PerceptralLoss, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, input, target):
        feature_in = self.vgg19_54(input)
        feature_tar = self.vgg19_54(target)
        loss = F.l1_loss(input=feature_in, target=feature_tar)
        return loss