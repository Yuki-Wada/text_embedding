"""
Define Utility Functions Related to Optimizers.
"""
from typing import Dict
from tensorflow.keras import optimizers

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

def get_keras_optimizer(optimizer_params: Dict):
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
