# coding: utf-8
import attr
# from warmup_scheduler import GradualWarmupScheduler
# from trainer.scheduler.warm_up import GradualWarmupScheduler

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR
# from torch_poly_lr_decay import PolynomialLRDecay

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer


@attr.s
class SchedulerSetting:
    name = attr.ib()
    optimizer = attr.ib()
    enable_warmup = attr.ib()
    max_epoch = attr.ib()
    max_iteration = attr.ib()
    warmup_multiplier = attr.ib()
    warmup_iteration = attr.ib()
    h_params = attr.ib()


def build_scheduler(setting: SchedulerSetting):

    name = setting.name
    optimizer = setting.optimizer
    enable_warmup = setting.enable_warmup
    warmup_multiplier = setting.warmup_multiplier
    warmup_iteration = setting.warmup_iteration

    if name == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        if not enable_warmup:
            return scheduler
        return GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_warmup_iterations=warmup_iteration,
                                      after_scheduler=scheduler)

    # elif name == "polynomial":
    #     scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=150, end_learning_rate=0.0001,
    #                                                     power=2.0)
    #     if not enable_warmup:
    #         return scheduler_poly_lr_decay
    #     return GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_warmup_iterations=warmup_iteration,
    #                                                       after_scheduler=scheduler_poly_lr_decay)
    elif name == 'linear':
        alpha = setting.h_params.get('alpha', 0.1)
        T = setting.h_params.get('T', 10)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: alpha if epoch >
                             T else 1.0 - (1.0 - alpha) * epoch / T)
        if not enable_warmup:
            return scheduler
        return GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_warmup_iterations=warmup_iteration,
                                      after_scheduler=scheduler)

    elif name == "cosine":
        scheduler_cosine = CosineAnnealingLR(optimizer, setting.max_iteration)
        if not enable_warmup:
            return scheduler_cosine
        return GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_warmup_iterations=warmup_iteration,
                                      after_scheduler=scheduler_cosine)

    else:
        scheduler_steplr = StepLR(optimizer, step_size=100, gamma=0.5)
        if not enable_warmup:
            return scheduler_steplr
        return GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_warmup_iterations=warmup_iteration,
                                      after_scheduler=scheduler_steplr)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_warmup_iterations: target learning rate is reached at total_warmup_iterations, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

        this class is based on following article
        https://blog.csdn.net/qq_40859461/article/details/93139855

        last_epoch (int): The index of last epoch. Default: -1.
        Probably, this value can be used as a last steps, cuz this value is incremented when step() is called
    """

    def __init__(self, optimizer, multiplier, total_warmup_iterations, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_warmup_iterations = total_warmup_iterations
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_warmup_iterations:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_warmup_iterations + 1.) for base_lr in self.base_lrs]

    def get_last_lr(self):
        return self.get_lr()

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(
                        epoch - self.total_warmup_iterations)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            raise NotImplementedError
