# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:16:33
# @FileName:  select_dataset.py
# @Contact :  lianghao@whu.edu.cn


class DataSet:
    def __init__(self, option):
        self.patch_size = 0
        self.dataset = option['dataset']
        self.task = option['dataset']['task']['name']
        self.mode = option['network']['mode']
        self.action = option['global_setting']['action']
        if self.action.upper() == 'train'.upper():
            self.patch_size = option['train']['patch']

    def __call__(self):

        if self.dataset['name'].upper() == 'basic_pair'.upper():
            train_dataset, test_dataset = None, None
            from dataset.basic_dataset.pair_real import TrainDataSet, TestDataSet
            if 'train' in self.dataset.keys():
                train_dataset = TrainDataSet(self.dataset['train']['input'],
                                             self.dataset['train']['target'], mode=self.mode)
            test_dataset = TestDataSet(self.dataset['test']['input'],
                                         self.dataset['test']['target'], mode=self.mode)
            return train_dataset, test_dataset

        if self.task.upper() == 'denoise'.upper():
            from dataset.denoise.select import select_dataset


        train_dataset, test_dataset = select_dataset(self.dataset, patch_size=self.patch_size, mode=self.mode)

        return train_dataset, test_dataset