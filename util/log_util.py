# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:17:20
# @FileName:  log_util.py
# @Contact :  lianghao@whu.edu.cn

import os
import shutil
import logging
import datetime


class Logger:
    def __init__(self, log_dir):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        log_name = os.path.join(log_dir, 'logger.log')
        log_file = logging.FileHandler(filename=log_name, mode='a', encoding='utf-8')
        log_file.setFormatter(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s %(asctime)s >> %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)

    def __call__(self):
        return self.logger


class Recorder:
    def __init__(self, option):
        self.option = option
        self.dir = option['directory']

    def __current_time__(self):
        dtime = datetime.datetime.now()
        timestr = str(dtime.month).zfill(2) + str(dtime.day).zfill(2) + '_' + str(dtime.hour).zfill(2) + '_' + str(
            dtime.minute).zfill(2) + '_' + str(dtime.second).zfill(2)
        return timestr

    def __check_dir__(self, *dirs):
        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

    def __call__(self):
        inlet_dir = self.option['global_setting']['record_dir']
        action_dir = os.path.join(inlet_dir, self.option['global_setting']['action'])
        task_dir = os.path.join(action_dir, self.option['global_setting']['task'])
        model_dir = os.path.join(task_dir, self.option['global_setting']['note_name'])
        self.main_record = os.path.join(model_dir, self.__current_time__())
        self.__check_dir__(inlet_dir, action_dir, task_dir, model_dir, self.main_record)
        for key in self.dir.keys():
            folder = os.path.join(self.main_record, self.dir[key])
            self.__check_dir__(folder)
        self.__copy_file__()

    def __copy_file__(self):
        if self.option['global_setting']['resume']['state']:
            state_file = self.option['global_setting']['resume']['checkpoint']
            _, name = os.path.split(state_file)
            save_dir = os.path.join(self.main_record, self.option['directory']['resume'])
            save_path = os.path.join(save_dir, name)
            shutil.copyfile(state_file, save_path)