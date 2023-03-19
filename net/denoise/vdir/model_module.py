import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_1(x) + x
        return x


class RIRblock(nn.Module):
    def __init__(self, in_ch, out_ch, num=4):
        super(RIRblock, self).__init__()
        self.resblock = []
        for _ in range(num):
            self.resblock.append(
                ResBlock(in_ch, in_ch,)
            )
        self.resblock = nn.Sequential(*self.resblock)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv(self.resblock(x)) + x
        return x

class Denoiser(nn.Module):
    def __init__(self, in_ch, out_ch, wf, num_res, num_rir):
        super(Denoiser, self).__init__()
        self.conv_begin = nn.Conv2d(in_ch, wf, 3, 1, 1)
        self.rirblock = []
        for _ in range(num_rir):
            self.rirblock.append(
                RIRblock(wf, wf, num=num_res)
            )
        self.conv_mid = nn.Conv2d(wf, wf, 3, 1, 1)
        self.conv_end = nn.Conv2d(wf, out_ch, 3, 1, 1)
        self.rirblock = nn.Sequential(*self.rirblock)

    def forward(self, y, c):
        condition = nn.UpsamplingBilinear2d(size=y.shape[2:])
        c = condition(c)
        x = torch.cat([y, c], dim=1)
        x1 = self.conv_begin(x)
        x2 = self.rirblock(x1)
        x2 = self.conv_mid(x2) + x1
        x3 = self.conv_end(x2) + y
        return x3




class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, wf):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, wf, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_5 = nn.Conv2d(wf, wf, 3, 1, 1)
        self.mu = nn.Conv2d(wf, out_ch, 3, 1, 1)
        self.sigma = nn.Conv2d(wf, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, wf):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, wf, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReLU(inplace=True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(wf, wf, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Conv2d(wf, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.last(x)
        return x


class Estimator(nn.Module):
    def __init__(self, in_ch, out_ch, wf):
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, wf, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Conv2d(wf, out_ch, 3, 1, 1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        out = self.up(x)
        return out


class SNConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, bias=True):
        super(SNConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        nn.utils.spectral_norm(self.conv)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, act=True):
        if act:
            x = self.conv(x)
            x = self.relu(x)
        else:
            x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch, wf):
        super(Discriminator, self).__init__()
        self.conv_1_1 = SNConv(in_ch, wf, 3, stride=1, padding=1)
        self.conv_1_2 = SNConv(wf, wf * 2, 3, stride=2, padding=1)

        self.conv_2_1 = SNConv(wf * 2, wf * 2, 3, stride=1, padding=1)
        self.conv_2_2 = SNConv(wf * 2, wf * 4, 3, stride=2, padding=1)

        self.conv_3_1 = SNConv(wf * 4, wf * 4, 3, stride=1, padding=1)
        self.conv_3_2 = SNConv(wf * 4, wf * 8, 3, stride=2, padding=1)

        self.conv_4_1 = SNConv(wf * 8, wf * 8, 3, stride=1, padding=1)
        self.conv_4_2 = SNConv(wf * 8, wf * 8, 3, stride=2, padding=1)

        self.conv_5_1 = SNConv(wf * 8, wf * 8, 3, stride=1, padding=1)
        self.conv_5_2 = SNConv(wf * 8, wf * 8, 3, stride=2, padding=1)

        self.last = SNConv(wf * 8, 1, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.conv_2_2(self.conv_2_1(x))
        x = self.conv_3_1(x)
        conv_3_2 = self.conv_3_2(x)
        x = self.conv_4_1(conv_3_2)
        conv_4_2 = self.conv_4_2(x)
        x = self.conv_5_1(conv_4_2)
        conv_5_2 = self.conv_5_2(x)
        # feature = [conv_3_2, conv_4_2, conv_5_2]
        logit = self.last(conv_5_2)
        return logit
        
