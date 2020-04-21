"""
Define Utility Functions Related to Optimizers.
"""
from typing import Dict
from torch.nn import Parameter
from torch.optim import Optimizer, SGD, Adadelta, Adam
from adabound import AdaBound
from tensorflow.keras import optimizers, callbacks

def get_torch_optimizer(
        model_weights: Dict[str, Parameter],
        optimizer_type: str,
        optimizer_params: Dict) -> Optimizer:

    if optimizer_type in ('SGD', 'MomentumSGD', 'NAG'):
        optimizer = SGD(
            model_weights,
            lr=optimizer_params['lr'],
            momentum=optimizer_params['momentum'],
            dampening=0,
            weight_decay=optimizer_params['weight_decay'],
            nesterov=optimizer_params['nesterov']
        )
    elif optimizer_type == 'Adadelta':
        optimizer = Adadelta(
            model_weights,
            lr=optimizer_params['lr'],
            rho=optimizer_params['rho'],
            weight_decay=optimizer_params['weight_decay']
        )
    elif optimizer_type == 'Adam':
        optimizer = Adam(
            model_weights,
            lr=optimizer_params['lr'],
            weight_decay=optimizer_params['weight_decay']
        )
    elif optimizer_type == 'AdaBound':
        optimizer = AdaBound(
            model_weights,
            lr=optimizer_params['lr'],
            final_lr=optimizer_params['final_lr'],
            weight_decay=optimizer_params['weight_decay']
        )
    else:
        raise ValueError('The optimizer {} is not supported.'.format(optimizer_type))

    return optimizer

class LearningRateCalculator:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, epoch):
        curr_lr = self.lr
        if epoch >= 150:
            curr_lr *= 0.01
        if epoch >= 225:
            curr_lr *= 0.001

        return curr_lr

def get_keras_optimizer(optimizer_params: Dict) -> Optimizer:
    if optimizer_params['optim'] == 'sgd':
        return optimizers.SGD(
            lr=optimizer_params['lr'],
            decay=optimizer_params['decay'],
            momentum=optimizer_params['momentum'],
            nesterov=optimizer_params['nesterov'],
        )
    if optimizer_params['optim'] == 'adadelta':
        return optimizers.Adadelta(
            decay=optimizer_params['decay']
        )
    if optimizer_params['optim'] == 'adam':
        return optimizers.Adam(
            lr=optimizer_params['lr'],
            decay=optimizer_params['decay']
        )

    raise ValueError('The optimizer {} is not supported.'.format(optimizer_params['optim']))
