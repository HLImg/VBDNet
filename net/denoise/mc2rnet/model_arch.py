import torch
import torch.nn as nn

class CrossConcatenation(nn.Module):
    def __init__(self, in_channels, wf):
        super(CrossConcatenation, self).__init__()
        self.up_one = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=wf, out_channels=wf, kernel_size=1, dilation=2, bias=True),
            nn.ReLU(inplace=True)
        ])
        self.up_two = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels * 2, out_channels=wf, kernel_size=1, dilation=2, bias=True),
            nn.ReLU(inplace=True)
        ])
        self.down_one = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])
        self.down_two = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])
        self.down_three = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])
    def forward(self, x):
        out_up = self.up_one(x) + x
        out_down = self.down_one(x)
        out_up = torch.cat([out_up, out_down], dim=1)
        out_up = self.up_two(out_up)
        out_down = self.down_two(out_down) + x
        out_down = self.down_three(out_down)
        return out_down, out_up

class MC2RNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, wf=64, depth=3):
        super(MC2RNet, self).__init__()
        in_channels = in_ch
        out_channles = out_ch
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])
        self.c2M = nn.ModuleList()
        for i in range(0, depth):
            self.c2M.append(
                CrossConcatenation(in_channels=wf, wf=wf)
            )
        self.conv2 = nn.Conv2d(in_channels=wf, out_channels=out_channles, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=wf, out_channels=out_channles, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out_conv = self.conv1(x)
        for idx, c2m in enumerate(self.c2M):
            out_up, out_down = c2m(out_conv)
            out_conv = out_up + out_down
        out_up = self.conv2(out_up)
        out_down = self.conv3(out_down)
        out = out_up + out_down + x
        return out