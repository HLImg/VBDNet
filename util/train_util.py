import os
import torch
import numpy as np
import torch.nn as nn
from thop import profile
import torch.nn.init as init
from timm.models.layers import trunc_normal_


def performance(self, net, input):
    macs, params = profile(self.net, (input, ), verbose=False)
    print(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    print(
        f'#                                     macs = {macs / 1e9:.6} G,  params = {params / 1e6 : 6} M                                              #')
    print(
        '# --------------------------------------------------------------------------------------------------------------------------#')


def resume_state(checkpoint, net, optimizer=None, scheduler=None, mode='all'):
    assert checkpoint is not None, "checkpoint is NOne"
    checkpoint = torch.load(checkpoint)
    if mode.upper() == 'all'.upper():
        next_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['model'], strict=False)
        if scheduler is not None:
            for _ in range(1, next_epoch):
                scheduler.step()
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        return next_epoch, net, optimizer, scheduler

    elif mode.upper() == 'model'.upper():
        net.load_state_dict(checkpoint['model'])
        return net
    else:
        if len(mode) == 0:
            net.load_state_dict(checkpoint)
        else:
            net.load_state_dict(checkpoint[mode])
        return net



class Early_Stop:
    def __init__(self, logger, patience=50, verbose=False, delta=0, save_dir=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_max = np.Inf
        self.val_metric_min = 0
        self.delta = delta
        self.save_dir = save_dir
        self.logger = logger

    def stop_loss(self, epoch, model, optimizer, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_state(epoch, model, optimizer, score=val_loss, valid_mode='loss')

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter : {self.counter} / {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_state(epoch, model, optimizer, score=val_loss, valid_mode='loss')
            self.best_score = score
            self.counter = 0

    def stop_metric(self, epoch, model, optimizer, val_metric):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_state(epoch, model, optimizer,  score=val_metric, valid_mode='loss')

        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.logger.info(f'EarlyStopping counter : {self.counter} / {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_state(epoch, model, optimizer, score=score, valid_mode='metric')
            self.best_score = score
            self.counter = 0

    def save_state(self, epoch, model, optimizer, score, valid_mode):
        if self.verbose:
            self.logger.info(
                f' epoch = {epoch}, validation ' + valid_mode + f' changed ({self.best_score:.6f} ---> {score:.6f}).')
        checkpoint = {
            'valid_mode': valid_mode,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'score': score
        }
        save_name = os.path.join(self.save_dir, f'model_current_{str(epoch).zfill(4)}.pth')
        torch.save(obj=checkpoint, f=save_name)



class WeightInit:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        if self.name.upper() == 'swinconv'.upper():
            return self.__swinconv__
        elif self.name.upper() == 'kaiming'.upper():
            return self.__kaiming__
        elif self.name.upper() == 'swinunet'.upper():
            return self.__swinunet__

    def __swinconv__(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __kaiming__(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __swinunet__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= 1  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                m, nn.ConvTranspose3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= 1  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(
                m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)