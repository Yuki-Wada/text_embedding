"""
Define Utility Functions Related to Optimizers.
"""
from typing import Dict

class ConstantLRScheduler:
    def __init__(self):
        pass

    def set_lr(self, optimizer, epoch): #pylint: disable=unused-argument
        return optimizer

class ExponentialDecayLRScheduler:
    def __init__(self, decay_rate, decay_epochs):
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs

    def set_lr(self, optimizer, epoch):
        if epoch in self.decay_epochs:
            optimizer.lr = optimizer.lr * self.decay_rate

        return optimizer

def get_lr_scheduler(scheduler_params):
    if scheduler_params['type'] == 'constant':
        return ConstantLRScheduler(**scheduler_params['kwargs'])
    if scheduler_params['type'] == 'exponential_decay':
        return ExponentialDecayLRScheduler(**scheduler_params['kwargs'])
    raise ValueError(
        'The learning rate scheduler {} is not supported.'.format(scheduler_params['type']))

def get_torch_optimizer(model_params, optimizer_params: Dict):
    import torch.optim as optim

    if 'decay' in optimizer_params['kwargs']:
        weight_decay = optimizer_params['kwargs']['decay']
        del optimizer_params['kwargs']['decay']
        optimizer_params['kwargs']['weight_decay'] = weight_decay

    if optimizer_params['type'] == 'sgd':
        return optim.SGD(
            model_params,
            **optimizer_params['kwargs']
        )
    if optimizer_params['type'] == 'adadelta':
        return optim.Adadelta(
            model_params,
            optimizer_params['kwargs']
        )
    if optimizer_params['type'] == 'adam':
        return optim.Adam(
            model_params,
            optimizer_params['kwargs']
        )

    raise ValueError('The optimizer {} is not supported.'.format(optimizer_params['type']))

def get_keras_optimizer(optimizer_params: Dict):
    from tensorflow.keras import optimizers

    lr_scheduler = get_lr_scheduler(optimizer_params['lr_scheduler'])
    if optimizer_params['type'] == 'sgd':
        return optimizers.SGD(**optimizer_params['kwargs']), lr_scheduler
    if optimizer_params['type'] == 'adadelta':
        return optimizers.Adadelta(**optimizer_params['kwargs']), lr_scheduler
    if optimizer_params['type'] == 'adam':
        return optimizers.Adam(**optimizer_params['kwargs']), lr_scheduler

    raise ValueError('The optimizer {} is not supported.'.format(optimizer_params['type']))
