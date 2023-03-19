# -*- coding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/03/19 21:19:43
# @FileName:  test.py
# @Contact :  lianghao@whu.edu.cn

import os
import shutil
from tqdm import tqdm
from util.log_util import Logger
from model.test.test_model import BasicModel
from util.log_util import Recorder

def test(model, logger):
    test_num = model.dataset_test.__len__()
    psnr, ssim, percep = 0.0, 0.0, 0.0
    with tqdm(total=test_num) as pbar:
        for i in range(0, test_num):
            _, file = os.path.split(model.dataset_test.input_paths[i])
            data = model.dataset_test.__getitem__(i)
            info = model.test(file, data)
            psnr += info[0]
            ssim += info[1]
            percep += info[2]
            pbar.set_description('testing ...')
            pbar.set_postfix(psnr=format(psnr / test_num, '.6f'), ssim=format(ssim / test_num, '.6f'), percep=format(percep / test_num, '.6f'))
            pbar.update(1)
            logger.info(f'{file}   ----- > psnr = {info[0]: .6f},  ssim = {info[1]: .6f},  perceptual = {info[2]: .6f}')

    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Finish Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        f'Metrics(average) : psnr = {psnr / test_num : .6f}, ssim = {ssim / test_num: .6f}, perceptural_loss = {percep / test_num : .6f}')


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
        '#                                                   Start Testing                                                           #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    test(model, logger)
    return True
