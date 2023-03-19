# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:20:02
# @FileName:  real.py
# @Contact :  lianghao@whu.edu.cn

import os
import shutil
from tqdm import tqdm
from util.log_util import Logger
from model.test.real_model import BasicModel
from util.log_util import Recorder


def realtest(model, logger):
    test_num = model.dataset_test.__len__()
    with tqdm(total=test_num) as pbar:
        pbar.set_description('running ...')
        for i in range(0, test_num):
            _, file = os.path.split(model.dataset_test.input_paths[i])
            data = model.dataset_test.__getitem__(i)
            model.test(file, data)
            pbar.update(1)

    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Finish Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')


def inlet(option, args):
    recorder = Recorder(option=option)
    recorder()
    _, yamlfile = os.path.split(args.yaml)
    shutil.copy(args.yaml, os.path.join(recorder.main_record, yamlfile))
    logger = Logger(log_dir=recorder.main_record)()
    model = BasicModel(option, logger, main_dir=recorder.main_record)
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   tart Testing                                                            #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    realtest(model, logger)
    return True
