# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:26:45
# @FileName:  EstimateNet.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn

class LDNet(nn.Module):
    """
    Encoder - Decoder
    """
    def __init__(self, inchannels, outchannels, wf, depth, slope):
        super(LDNet, self).__init__()
        self.depth = depth
        self.down = nn.ModuleList()
        preview = inchannels
        for i in range(self.depth):
            self.down.append(
                ConvBlock(inchannel=preview, outchannel=wf, slope=slope)
            )
            self.down.append(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            )
            preview = wf

        self.center = ConvBlock(inchannel=preview, outchannel=wf, slope=slope)

        self.up = nn.ModuleList()
        for i in range(self.depth):
            self.up.append(
                nn.ConvTranspose2d(in_channels=wf, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.up.append(
                ConvBlock(inchannel=wf, outchannel=wf, slope=slope)
            )
        self.last = nn.Conv2d(in_channels=wf, out_channels=outchannels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        for i, down in enumerate(self.down):
            x = down(x)
        x = self.center(x)
        for i, up in enumerate(self.up):
            x = up(x)
        out = self.last(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel, slope):
        super(ConvBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=slope, inplace=True)
        )

    def forward(self, x):
        return self.down(x)