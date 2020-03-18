from typing import Dict
from torch.nn import Parameter
from torch.optim import Optimizer, SGD, Adadelta, Adam
from adabound import AdaBound

def get_optimizer(
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
