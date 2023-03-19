import torch.nn as nn
from net.denoise.vdn.UNet import UNet
from net.denoise.vdn.DnCNN import DnCNN

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net


class VDN(nn.Module):
    def __init__(self, in_ch, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        self.in_ch = in_ch
        self.DNet = UNet(in_ch, in_ch*2, wf=wf, depth=dep_U, slope=slope)
        self.SNet = DnCNN(in_ch, in_ch*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='test'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            im_denoise = x - phi_Z[:, :1, ].detach().data
            im_denoise.clamp_(0.0, 1.0)
            return im_denoise
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma