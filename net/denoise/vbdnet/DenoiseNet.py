# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:26:19
# @FileName:  DenoiseNet.py
# @Contact :  lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
    """
    @inproceedings{ronneberger2015u,
        title={U-net: Convolutional networks for biomedical image segmentation},
        author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
        booktitle={International Conference on Medical image computing and computer-assisted intervention},
        pages={234--241},
        year={2015},
        organization={Springer}
        }
    """

    def __init__(self, in_channel=1, out_channel=2, depth=4, wf=64, slope=0.2):
        super(UNET, self).__init__()
        self.depth = depth
        prev_inchannel = in_channel
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            self.down_path.append(
                ConvBlock(in_channel=prev_inchannel, out_channel=(2 ** i) * wf, slope=slope)
            )
            prev_inchannel = (2 ** i) * wf
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(
                UpConvBlock(in_channel=prev_inchannel, out_channel=(2 ** i) * wf, slope=slope)
            )
            prev_inchannel = (2 ** i) * wf

        self.last = nn.Conv2d(in_channels=prev_inchannel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        block_down = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                block_down.append(x)
                x = F.avg_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, block_down[-i - 1])
        out = self.last(x)
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, slope=0.2):
        super(ConvBlock, self).__init__()
        block = [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, slope=0.2):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=True)
        self.conv_block = ConvBlock(in_channel=in_channel, out_channel=out_channel, slope=slope)

    def forward(self, x, bridge):
        up_right = self.up(x)
        crop_left = self.crop(bridge, up_right.shape[2:])
        cat_tensor = torch.cat([crop_left, up_right], 1)
        out = self.conv_block(cat_tensor)
        return out

    def crop(self, layer, target_size):
        _, _, layerH, layerW = layer.size()
        difH = (layerH - target_size[0]) // 2
        difW = (layerW - target_size[1]) // 2
        elayer = layer[:, :, difH : (difH + target_size[0]), difW : (difW + target_size[1])]
        return elayer