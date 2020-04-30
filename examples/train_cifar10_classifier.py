"""
Train a classifier for CIFAR-10 data set.
"""
import os
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from mltools.utils import set_tensorflow_gpu, set_tensorflow_seed, set_logger, \
    dump_json, get_date_str
from mltools.dataset.cifar10 import Cifar10DataSet, Cifar10DataLoader
from mltools.model.image_classification import ConvNet, ResNet, calc_loss
from mltools.optimizer.utils import get_keras_optimizer
from mltools.metric.metric_manager import MerticManager

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=-1)

    parser.add_argument("--train_image_npy_path", help="train image npy file", required=True)
    parser.add_argument("--train_label_npy_path", help="train label npy file", required=True)
    parser.add_argument("--test_image_npy_path", help="test image npy file", required=True)
    parser.add_argument("--test_label_npy_path", help="test label npy file", required=True)
    parser.add_argument("--output_dir_format", default='.')
    parser.add_argument('--model_name_format', default='epoch-{epoch}.hdf5')

    parser.add_argument('--model', default='conv_net')

    parser.add_argument('--optimizer', dest='optim', default='sgd')
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--clipvalue', type=float)
    parser.add_argument('--clipnorm', type=float)

    parser.add_argument('--lr_scheduler', default='exponential_decay')
    parser.add_argument('--lr_decay_rate', type=float, default=1e-1)
    parser.add_argument('--lr_decay_epochs', nargs='+', type=float, default=[3, 16, 23, 28])

    parser.add_argument('--epochs', help="epoch count", type=int, default=256)
    parser.add_argument('--mb_size', help="minibatch size", type=int, default=64)

    parser.add_argument('--seed', type=int, help='random seed for initialization')

    args = parser.parse_args()

    return args

def get_model_params(args):
    return {
        'model': args.model,
        'label_count': 10,
        'input_shape': (32, 32, 3),
    }

def get_model(model_params):
    if model_params['model'] == 'conv_net':
        return ConvNet(
            label_count=model_params['label_count'],
            input_shape=model_params['input_shape'],
        )
    if model_params['model'] == 'res_net':
        return ResNet(
            label_count=model_params['label_count'],
            input_shape=model_params['input_shape'],
        )
    raise ValueError('The model {} is not supported.'.format(model_params['model']))

def get_optimizer_params(args):
    lr_scheduler_params = {}
    lr_scheduler_params['type'] = args.lr_scheduler
    lr_scheduler_params['kwargs'] = {}
    if args.lr_scheduler == 'constant':
        pass
    if args.lr_scheduler == 'exponential_decay':
        lr_scheduler_params['kwargs']['decay_rate'] = args.lr_decay_rate
        lr_scheduler_params['kwargs']['decay_epochs'] = args.lr_decay_epochs
    else:
        raise ValueError(
            'The learning rate scheduler {} is not supported.'.format(args.lr_scheduler))

    optimizer_params = {}
    optimizer_params['type'] = args.optim
    optimizer_params['lr_scheduler'] = lr_scheduler_params

    optimizer_params['kwargs'] = {}
    if args.clipvalue:
        optimizer_params['kwargs']['clipvalue'] = args.clipvalue
    if args.clipnorm:
        optimizer_params['kwargs']['clipnorm'] = args.clipnorm

    if args.optim == 'sgd':
        optimizer_params['kwargs']['lr'] = args.lr
        optimizer_params['kwargs']['decay'] = args.lr * args.weight_decay
        optimizer_params['kwargs']['momentum'] = args.momentum
        optimizer_params['kwargs']['nesterov'] = args.nesterov

        return optimizer_params

    if args.optim == 'adadelta':
        optimizer_params['kwargs']['decay'] = args.weight_decay

        return optimizer_params

    if args.optim == 'adam':
        optimizer_params['kwargs']['lr'] = args.lr
        optimizer_params['kwargs']['decay'] = args.weight_decay

        return optimizer_params

    raise ValueError('The optimizer {} is not supported.'.format(args.optimizer))

def setup_output_dir(output_dir_path, args, model_params, optimizer_params):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))
    dump_json(model_params, os.path.join(output_dir_path, 'model.json'))
    dump_json(optimizer_params, os.path.join(output_dir_path, 'optimizer.json'))

def get_cm(trues, preds, label_count):
    cm = np.zeros((label_count, label_count), dtype=np.int32)
    for true, pred in zip(trues, preds):
        cm[true, pred] += 1
    return cm

def train(
        output_dir_path,
        model_name_format,
        train_data_loader,
        valid_data_loader,
        model_params,
        optimizer_params,
        epochs,
        best_monitored_metric=None,
        seed=None,
    ):
    set_tensorflow_seed(seed)

    # Set up Model and Optimizer
    model = get_model(model_params)
    model.summary()
    optimizer, lr_scheduler = get_keras_optimizer(optimizer_params)

    # Train Model
    mertic_manager = MerticManager(output_dir_path, epochs)
    for epoch in range(epochs):
        logger.info('Epoch: %d', epoch + 1)

        # Train
        lr_scheduler.set_lr(optimizer, epoch + 1)

        train_loss_sum = 0
        train_data_count = 0
        train_cm = np.zeros((10, 10), dtype=np.int32)
        with tqdm(total=len(train_data_loader), desc="Train CNN") as pbar:
            for mb_images, mb_labels in train_data_loader:
                mb_count = mb_images.shape[0]
                try:
                    with tf.GradientTape() as tape:
                        mb_probs = model(mb_images, training=True)
                        mb_train_loss = calc_loss(mb_labels, mb_probs)

                    grads = tape.gradient(mb_train_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    mb_train_loss = mb_train_loss.numpy()
                    mb_preds = np.argmax(mb_probs.numpy(), axis=1)
                    mb_cm = get_cm(mb_labels, mb_preds, 10)

                    train_loss_sum += mb_train_loss * mb_count
                    train_cm += mb_cm
                    train_data_count += mb_count

                except RuntimeError as error:
                    logger.error(str(error))
                    mb_train_loss = np.nan

                finally:
                    pass

                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=mb_train_loss,
                    accu=np.sum(mb_cm * np.eye(10)) / mb_count,
                ))

            train_loss = train_loss_sum / train_data_count
            train_accuracy = np.sum(train_cm * np.eye(10)) / train_data_count
            logger.info('Train Loss: %f', train_loss)
            logger.info('Train Accuracy: %f', train_accuracy)
            mertic_manager.register_loss(train_loss, epoch, 'train')

        # Valid
        valid_loss_sum = 0.0
        valid_data_count = 0
        valid_cm = np.zeros((10, 10), dtype=np.int32)
        with tqdm(total=len(valid_data_loader), desc='Valid') as pbar:
            for mb_images, mb_labels in valid_data_loader:
                mb_count = mb_images.shape[0]

                try:
                    mb_probs = model(mb_images, training=False)
                    mb_valid_loss = calc_loss(mb_labels, mb_probs).numpy()

                    mb_preds = np.argmax(mb_probs.numpy(), axis=1)
                    mb_cm = get_cm(mb_labels, mb_preds, 10)

                    valid_loss_sum += mb_valid_loss * mb_count
                    valid_cm += mb_cm
                    valid_data_count += mb_count

                except Exception as error:
                    logger.error(str(error))
                    mb_valid_loss = np.nan

                finally:
                    pass

                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=mb_valid_loss,
                    accu=np.sum(mb_cm * np.eye(10)) / mb_count,
                ))

            valid_loss = valid_loss_sum / valid_data_count
            valid_accuracy = np.sum(valid_cm * np.eye(10)) / valid_data_count
            logger.info('Valid Loss: %f', valid_loss)
            logger.info('Valid Accuracy: %f', valid_accuracy)
            mertic_manager.register_loss(valid_loss, epoch, 'valid')

        # Save
        monitored_metric = - valid_loss
        if best_monitored_metric is None or best_monitored_metric < monitored_metric:
            best_monitored_metric = monitored_metric
            logger.info('The current score is best.')
            if model_name_format:
                model_name = model_name_format.format(epoch=epoch + 1)
                logger.info('Save the model as %s', model_name)
                model.save_weights(os.path.join(output_dir_path, model_name))

    return best_monitored_metric

def run():
    set_logger()
    set_tensorflow_gpu()
    args = get_args()
    set_tensorflow_seed(args.seed)

    train_data_set = Cifar10DataSet(args.train_image_npy_path, args.train_label_npy_path)
    train_data_loader = Cifar10DataLoader(train_data_set, args.mb_size)
    test_data_set = Cifar10DataSet(args.test_image_npy_path, args.test_label_npy_path)
    test_data_loader = Cifar10DataLoader(test_data_set, args.mb_size)

    model_params = get_model_params(args)
    optimizer_params = get_optimizer_params(args)

    output_dir_path = args.output_dir_format.format(date=get_date_str())
    setup_output_dir(output_dir_path, dict(args._get_kwargs()), model_params, optimizer_params) #pylint: disable=protected-access

    train(
        output_dir_path=output_dir_path,
        model_name_format=args.model_name_format,
        train_data_loader=train_data_loader,
        valid_data_loader=test_data_loader,
        model_params=model_params,
        optimizer_params=optimizer_params,
        epochs=args.epochs,
        seed=args.seed,
    )

if __name__ == '__main__':
    run()
