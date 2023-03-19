# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:27:06
# @FileName:  FusionNet.py
# @Contact :  lianghao@whu.edu.cn

import torch.nn as nn
from net.denoise.vbdnet.DenoiseNet import UNET
from net.denoise.vbdnet.EstimateNet import LDNet
from util.data_util import padding

class VBDNet(nn.Module):
    """
    VDN Architecture
    """
    def __init__(self, in_ch, wf=64, depth_D=4, depth_S=5, slope=0.2):
        super(VBDNet, self).__init__()
        self.in_ch = in_ch
        self.dnet = UNET(in_channel=in_ch, out_channel=in_ch * 2, depth=depth_D, slope=slope)
        self.snet = LDNet(inchannels=in_ch, outchannels=in_ch * 2, wf=wf, depth=depth_S, slope=slope)

    def forward(self, x, mode='test'):
        if mode.lower() == 'train':
            z = self.dnet(x)
            sigma = self.snet(x)
            return z, sigma
        elif mode.lower() == 'test':
            z = self.dnet(x)
            z = z[:, :self.in_ch, ]
            return z
        elif mode.lower() == 'sigma':
            sigma = self.snet(x)
            return sigma