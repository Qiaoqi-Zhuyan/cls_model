import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train_epoches * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epoches * n_iter_per_epoch)
    decay_steps = int(config.decay_epoches * n_iter_per_epoch)

    if config.use_cosine:
        lr_scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial= num_steps,
            lr_min=config.mix_lr,
            warmup_lr_init=config.warmup_lr
        )
        return lr_scheduler
