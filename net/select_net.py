# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:20:39
# @FileName:  select_net.py
# @Contact :  lianghao@whu.edu.cn


class Net:
    def __init__(self, option):
        self.task = option['network']['task']
        self.info_net = option['network']

    def __call__(self):
        if self.task.upper() == 'denoise'.upper():
            from net.denoise.select import select_network

        network = select_network(self.info_net)
        return network