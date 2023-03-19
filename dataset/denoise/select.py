# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:31:13
# @FileName:  select.py
# @Contact :  lianghao@whu.edu.cn

import numpy as np

def select_dataset(info_dataset, patch_size, mode):
    clip = info_dataset['clip']
    name = info_dataset['name']
    task = info_dataset['task']['noise']
    train_dataset, test_dataset = (None, None)

    if name.upper() == 'unpair'.upper():
        from dataset.denoise.denoise_unpair import TrainDataSet, TestDataSet
        if 'train' in info_dataset.keys():
            train_noise_levels = get_noise_level(info_dataset, action='train')
            train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], patch_size=patch_size,
                                         levels=train_noise_levels, task=task,
                                         mode=mode, clip=clip)

        test_noise_levels = get_noise_level(info_dataset, action='test')
        test_dataset = TestDataSet(input_dir=info_dataset['test']['target'], levels=test_noise_levels,
                                   task=task, mode=mode, clip=clip)
    elif name.upper() == 'vdn'.upper():
        from dataset.denoise.vdn import TrainDataSet, TestDataSet
        if 'train' in info_dataset.keys():
            train_noise_levels = get_noise_level(info_dataset, action='train')
            train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], patch_size=patch_size,
                                         levels=train_noise_levels, task=task,
                                         mode=mode, clip=clip)
        test_noise_levels = get_noise_level(info_dataset, action='test')
        test_dataset = TestDataSet(input_dir=info_dataset['test']['target'], levels=test_noise_levels,
                                   task=task, mode=mode, clip=clip)

    return train_dataset, test_dataset


def get_noise_level(info_dataset, action='train'):
    if isinstance(info_dataset[action]['levels'], list):
        noise_levels = info_dataset[action]['levels']
    else:
        noise_levels = np.arange(info_dataset[action]['levels']['begin'],
                                 info_dataset[action]['levels']['end'] + info_dataset[action]['levels']['step'],
                                 info_dataset[action]['levels']['step'])
    return noise_levels
