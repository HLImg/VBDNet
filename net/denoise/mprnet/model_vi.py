# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:23:12
# @FileName:  model_vi.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
from net.denoise.mprnet.model_arch import MPRNet
from net.denoise.vbdnet.EstimateNet import LDNet


class VIMPRNet(nn.Module):
    def __init__(self, in_ch):
        super(VIMPRNet, self).__init__()
        self.in_ch = in_ch
        self.dnet = MPRNet(in_ch=in_ch, out_ch=in_ch * 2, n_feat=80, bias=False)
        self.snet = LDNet(in_ch, in_ch * 2, wf=64, depth=5, slope=0.2)
        
    
    
    def forward(self, x, mode='test'):
        stage_3, stage_2, stage_1 = self.dnet(x, 'train')
        if mode.lower() == 'train':
            sigma = self.snet(x)
            return stage_3, stage_2, stage_1, sigma
        else:
            return stage_3[:, :self.in_ch, ]
