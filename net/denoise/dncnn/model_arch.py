import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_ch, out_ch, wf, depth):
        super(DnCNN, self).__init__()
        in_channels = in_ch
        self.firstLayer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        midLayer = []
        for i in range(1, depth - 1) :
            midLayer.append(nn.Conv2d(in_channels=wf, out_channels=wf, kernel_size=3, stride=1, padding=1, bias=True))
            # BN Layer
            midLayer.append(nn.BatchNorm2d(num_features=wf))
            midLayer.append(nn.ReLU(inplace=True))
        self.midLayer = nn.Sequential(*midLayer)
        self.lastLayer = nn.Conv2d(in_channels=wf, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = self.firstLayer(x)
        x1 = self.midLayer(x1)
        out = self.lastLayer(x1) + x
        return out