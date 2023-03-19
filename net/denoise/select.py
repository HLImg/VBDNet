# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:28:05
# @FileName:  select.py
# @Contact :  lianghao@whu.edu.cn

def select_network(info_net):
    name = info_net['name']
    params = info_net['params']

    if name.upper() == 'vdn'.upper():
        from net.denoise.vdn.model_arch import VDN as Net
    elif name.upper() == 'vbdnet'.upper():
        from net.denoise.vbdnet.FusionNet import VBDNet as Net
    elif name.upper() == 'vdir'.upper():
        from net.denoise.vdir.model_arch import VDIR as Net
    elif name.upper() == 'mc2rnet'.upper():
        from net.denoise.mc2rnet.model_arch import MC2RNet as Net
    elif name.upper() == 'dncnn'.upper():
        from net.denoise.dncnn.model_arch import DnCNN as Net
    elif name.upper() == 'mprnet'.upper():
        from net.denoise.mprnet.model_arch import MPRNet as Net    
    elif name.upper() == 'vimprnet'.upper():
        from net.denoise.mprnet.model_vi import VIMPRNet as Net
        
    network = Net(**params)
    return network
