"""
Train an Encoder-Decoder model.
"""
import os
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import tensorflow as tf

from mltools.utils import set_tensorflow_seed, set_logger, dump_json, get_date_str
from mltools.metric.metric_manager import MerticManager
from mltools.model.encoder_decoder import NaiveSeq2Seq, Seq2SeqWithGlobalAttention, decoder_loss
from mltools.optimizer.utils import get_keras_optimizer
from mltools.dataset.japanese_english_bilingual_corpus \
    import BilingualDataSet as DataSet, BilingualDataLoader as DataLoader

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', nargs='+', required=True)
    parser.add_argument('--valid_data', nargs='+', required=True)
    parser.add_argument('--output_dir_format', default='.')
    parser.add_argument('--model_name_format')

    parser.add_argument('--model', default='naive')
    parser.add_argument('--embedding_dimension', dest='emb_dim', type=int, default=400)
    parser.add_argument('--encoder_hidden_dimension', dest='enc_hidden_dim', type=int, default=200)
    parser.add_argument('--decoder_hidden_dimension', dest='dec_hidden_dim', type=int, default=200)

    parser.add_argument('--optimizer', dest='optim', default='adam')
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--final_lr', type=float, default=1e-1)

    parser.add_argument('--epochs', type=int, default=20, help='epoch count')
    parser.add_argument('--mb_size', type=int, default=32, help='minibatch size')

    parser.add_argument('--seed', type=int, help='random seed for initialization')

    args = parser.parse_args()

    return args

def get_model_params(args):
    return {
        'model': args.model,
        'emb_dim': args.emb_dim,
        'enc_hidden_dim': args.enc_hidden_dim,
        'dec_hidden_dim': args.dec_hidden_dim,
    }

def get_model(model_params):
    if model_params['model'] == 'naive':
        return NaiveSeq2Seq(
            model_params['ja_vocab_count'],
            model_params['en_vocab_count'],
            model_params['emb_dim'],
            model_params['enc_hidden_dim'],
            model_params['dec_hidden_dim'],
        )
    if  model_params['model'] == 'global_attention':
        return Seq2SeqWithGlobalAttention(
            model_params['ja_vocab_count'],
            model_params['en_vocab_count'],
            model_params['emb_dim'],
            model_params['enc_hidden_dim'],
            model_params['dec_hidden_dim'],
        )
    raise ValueError('The optimizer {} is not supported.'.format(model_params['model']))

def get_optimizer_params(args):
    if args.optim == 'sgd':
        return {
            'optim': args.optim,
            'lr': args.lr,
            'decay': args.lr * args.weight_decay,
            'momentum': args.momentum,
            'nesterov': args.nesterov,
        }

    if args.optim == 'adadelta':
        return {
            'optim': args.optim,
            'decay': args.weight_decay,
        }

    if args.optim == 'adam':
        return {
            'optim': args.optim,
            'lr': args.lr,
            'decay': args.weight_decay,
        }

    raise ValueError('The optimizer {} is not supported.'.format(args.optimizer))

def setup_output_dir(output_dir_path, args, model_params, optimizer_params):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))
    dump_json(model_params, os.path.join(output_dir_path, 'model.json'))
    dump_json(optimizer_params, os.path.join(output_dir_path, 'optimizer.json'))

def plot_metrics(metric_dict, label, figure_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for mode, metric in metric_dict.items():
        ax.plot(np.arange(len(metric)) + 1, metric, label=mode)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.legend()

    x_ax = ax.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(figure_path)
    plt.close()

def train_encoder_decoder(
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
    optimizer = get_keras_optimizer(optimizer_params)

    # Train Model
    mertic_manager = MerticManager(output_dir_path, epochs)
    for epoch in range(epochs):
        logger.info('Start Epoch %s', epoch + 1)

        # Train
        train_loss_sum = 0.0
        train_data_count = 0

        with tqdm(total=len(train_data_loader), desc='Train') as pbar:
            for mb_inputs, mb_outputs in train_data_loader:
                mb_count = mb_inputs.shape[0]

                with tf.GradientTape() as tape:
                    mb_probs = model(mb_inputs, mb_outputs[:, :-1], training=True)
                    mb_train_loss = decoder_loss(mb_outputs[:, 1:], mb_probs)

                grads = tape.gradient(mb_train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss_sum += mb_train_loss.numpy() * mb_count
                train_data_count += mb_count

                train_loss = train_loss_sum / train_data_count
                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=train_loss,
                    plex=np.exp(train_loss),
                ))
            mertic_manager.register_loss(train_loss, epoch, 'train')

        # Valid
        valid_loss_sum = 0.0
        valid_data_count = 0
        with tqdm(total=len(valid_data_loader), desc='Valid') as pbar:
            for mb_inputs, mb_outputs in valid_data_loader:
                mb_count = mb_inputs.shape[0]

                mb_probs = model(mb_inputs, mb_outputs[:, :-1])
                mb_valid_loss = decoder_loss(mb_outputs[:, 1:], mb_probs).numpy()

                valid_loss_sum += mb_valid_loss * mb_count
                valid_data_count += mb_count

                valid_loss = valid_loss_sum / valid_data_count
                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=valid_loss,
                    plex=np.exp(valid_loss),
                ))
            mertic_manager.register_loss(valid_loss, epoch, 'valid')

        # Save
        mertic_manager.plot_loss('Loss', os.path.join(output_dir_path, 'loss.png'))
        mertic_manager.save_score()

        monitored_metric = - valid_loss
        if best_monitored_metric is None or best_monitored_metric < monitored_metric:
            logger.info('The current score is best.')
            if model_name_format:
                model_name = model_name_format.format(epoch=epoch + 1)
                logger.info('Save the model as %s', model_name)
                model.save_weights(os.path.join(output_dir_path, model_name))

def run():
    set_logger()
    args = get_args()

    model_params = get_model_params(args)
    optimizer_params = get_optimizer_params(args)

    output_dir_path = args.output_dir_format.format(date=get_date_str())
    setup_output_dir(output_dir_path, dict(args._get_kwargs()), model_params, optimizer_params) #pylint: disable=protected-access

    train_data_set = DataSet(is_training=True)
    train_data_set.input_data(args.train_data)
    train_data_loader = DataLoader(train_data_set, args.mb_size)

    valid_data_set = DataSet(train_data_set.preprocessor, is_training=False)
    valid_data_set.input_data(args.valid_data)
    valid_data_loader = DataLoader(valid_data_set, args.mb_size)

    model_params['ja_vocab_count'] = train_data_set.ja_vocab_count
    model_params['en_vocab_count'] = train_data_set.en_vocab_count

    train_encoder_decoder(
        output_dir_path=output_dir_path,
        model_name_format=args.model_name_format,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        model_params=model_params,
        optimizer_params=optimizer_params,
        epochs=args.epochs,
        seed=args.seed,
    )

if __name__ == '__main__':
    run()
