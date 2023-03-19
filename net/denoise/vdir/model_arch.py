import torch
import torch.nn as nn
from util.data_util import padding
from net.denoise.vdir.model_module import *


class VDIR(nn.Module):
    def __init__(self, in_ch, out_ch, feat, wf, num_rir, num_res, gpu=True):
        super(VDIR, self).__init__()
        self.gpu = gpu
        self.encoder = Encoder(in_ch=in_ch, out_ch=feat, wf=wf)
        self.denoiser = Denoiser(in_ch=in_ch + feat, out_ch=out_ch, wf=wf, num_rir=num_rir, num_res=num_res)
        self.decoder = Decoder(in_ch=feat, out_ch=out_ch, wf=wf)
    def forward(self, x, mode='test'):
        mu, sigma = self.encoder(x)
        eps = torch.normal(mean=0, std=1, size=mu.shape)
        if self.gpu:
            eps = eps.cuda()
        c = mu + eps * torch.exp(sigma / 2)
        denoise = self.denoiser(x, c)
        dec = self.decoder(c)
        if mode == 'train':
            return denoise, dec, mu, sigma
        else:
            return denoise



if __name__ == '__main__':
    model = VDIR(1, 1, 4, wf=64, num_rir=5, num_res=5)
    a = torch.randn(1, 1, 320, 480)
    denoise, dec = model(a)
    print(denoise.shape, dec.shape)