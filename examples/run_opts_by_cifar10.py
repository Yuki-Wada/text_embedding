import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from mltools.utils import set_seed, set_logging_handler, setup_output_dir
from mltools.dataset.cifar10 import Cifar10DataSet, Cifar10DataLoader
from mltools.model.cnn import TempCNN
from mltools.optimizer.util import get_optimizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--use_cuda', dest='use_cuda', help='Use CUDA or not', action='store_true')

parser.add_argument(
    "--train_image_npy_path", type=str, help="input train image npy file", required=True)
parser.add_argument(
    "--train_label_npy_path", type=str, help="input train label npy file", required=True)
parser.add_argument(
    "--test_image_npy_path", type=str, help="input test image npy file", required=True)
parser.add_argument(
    "--test_label_npy_path", type=str, help="input test label npy file", required=True)
parser.add_argument("--output_dir", type=str, help="output directory path", required=True)

parser.add_argument('--optimizer', dest='optimizer', help='Optimizer', default='SGD')
parser.add_argument(
    '--lr', dest='lr', help='Learning rate of the optimizer', type=float, default=1e-4)
parser.add_argument(
    '--final_lr', dest='final_lr', help='Final learning rate of the optimizer for AdaBound',
    type=float, default=1e-1)
parser.add_argument(
    '--momentum', help='momentum of the optimizer for MomentumSGD or NAG',
    type=float, default=0.9)
parser.add_argument(
    '--rho',
    help='coefficient used for computing a running average of squared gradients for Adadelta',
    type=float, default=0.9)
parser.add_argument(
    '--weight_decay', dest='weight_decay', help='weight decay of the optimizer',
    type=float, default=0)
parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=10.0)
parser.add_argument(
    '--lr_trigger', dest='lr_trigger', nargs='+',
    help='The finished ratio at which to exponentially shift a learning rate',
    type=int, default=[2, 128, 192, 232])

parser.add_argument('--epochs', help="epoch count", type=int, default=256)
parser.add_argument('--mb_size', help="minibatch size", type=int, default=64)
parser.add_argument('--seed', help='random seed for initialization')

args = parser.parse_args()

def get_model_params():
    return {}

def get_optimizer_params():
    if args.optimizer == 'SGD':
        return {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': 0,
            'nesterov': False,
        }

    if args.optimizer == 'MomentumSGD':
        return {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'nesterov': False,
        }

    if args.optimizer == 'NAG':
        return {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'nesterov': True,
        }

    if args.optimizer == 'Adadelta':
        return {
            'lr': args.lr,
            'rho': args.rho,
            'weight_decay': args.weight_decay,
        }

    if args.optimizer == 'Adam':
        return {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }

    if args.optimizer == 'AdaBound':
        return {
            'lr': args.lr,
            'final_lr': args.final_lr,
            'weight_decay': args.weight_decay,
        }

    raise ValueError('The optimizer {} is not supported.'.format(args.optimizer))

def calc_loss():
    self.label_activate = nn.LogSoftmax(dim=1)
    self.label_loss = nn.NLLLoss()

def train(train_data_loader, test_data_loader, model, optimizer, epochs, output_dir_path):
    best_score = None
    for epoch in range(epochs):
        logger.info('Epoch: %d', epoch + 1)

        accum_train_loss = 0
        with tqdm(total=len(train_data_loader), desc="Train CNN") as pbar:
            for mb_images, mb_labels in train_data_loader.get_iter():
                curr_mb_size = mb_images.shape[0]
                try:
                    model.zero_grad()
                    outputs = model(mb_images)
                    loss = nn.NLLLoss()(nn.LogSoftmax(dim=1)(outputs), mb_labels)
                    loss.backward()
                    optimizer.step()

                    if loss.device.type == 'cuda':
                        loss = loss.cpu()
                    mb_average_loss = loss.data.numpy()
                    accum_train_loss += mb_average_loss * curr_mb_size
                    pbar.set_postfix(Loss='{:.4f}'.format(mb_average_loss))

                except RuntimeError as error:
                    logger.error(str(error))
                    mb_average_loss = np.nan

                finally:
                    if args.use_cuda:
                        torch.cuda.empty_cache()

                pbar.update(curr_mb_size)
                pbar.set_postfix(Loss='{:.4f}'.format(mb_average_loss))

        train_loss = accum_train_loss / len(train_data_loader)
        print(train_loss)

        curr_score = - train_loss
        if best_score is None or best_score < curr_score:
            logger.info('The current epoch score is best for the current parameter tuning.')
            best_score = curr_score
            torch.save(
                model.state_dict(),
                os.path.join(output_dir_path, 'cifar10_cnn.torch_params'))

def run():
    set_seed(args.seed)
    set_logging_handler([logger])
    output_dir_path = setup_output_dir(args.output_dir, dict(args._get_kwargs())) # pylint: disable=protected-access

    train_data_set = Cifar10DataSet(args.train_image_npy_path, args.train_label_npy_path)
    train_data_loader = Cifar10DataLoader(train_data_set, args.mb_size)
    test_data_set = Cifar10DataSet(args.test_image_npy_path, args.test_label_npy_path)
    test_data_loader = Cifar10DataLoader(test_data_set, args.mb_size)

    model_params = get_model_params()
    model = TempCNN(10)
    if args.use_cuda:
        model.cuda()

    optimizer_params = get_optimizer_params()
    optimizer = get_optimizer(model.parameters(), args.optimizer, optimizer_params)

    train(train_data_loader, test_data_loader, model, optimizer, args.epochs, output_dir_path)

if __name__ == '__main__':
    run()
