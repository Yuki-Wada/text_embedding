"""
Train an Encoder-Decoder model.
"""
import os
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import torch

from mltools.utils import set_seed, set_logger, dump_json, get_date_str
# from mltools.dataset.japanese_english_bilingual_corpus import \
#     BilingualDataSet as DataSet, BilingualDataLoader as DataLoader
from mltools.dataset.tanaka_corpus import \
    TanakaCorpusDataSet as DataSet, TanakaCorpusDataLoader as DataLoader
from mltools.model.encoder_decoder import decoder_loss, \
    NaiveSeq2Seq, Seq2SeqWithGlobalAttention, TransformerEncoderDecoder
from mltools.optimizer.utils import get_torch_optimizer, get_torch_lr_scheduler
from mltools.metric.metric_manager import MetricManager

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=-1)

    parser.add_argument('--train_data', nargs='+', required=True)
    parser.add_argument('--valid_data', nargs='+', required=True)
    parser.add_argument('--lang', default='ja_to_en')
    parser.add_argument('--output_dir_format', default='.')
    parser.add_argument('--model_name_format', default='epoch-{epoch}.hdf5')
    parser.add_argument('--preprocessor', default='preprocessor.bin')

    parser.add_argument('--model', default='naive')
    parser.add_argument('--embedding_dimension', dest='emb_dim', type=int, default=400)
    parser.add_argument('--encoder_hidden_dimension', dest='enc_hidden_dim', type=int, default=200)
    parser.add_argument('--decoder_hidden_dimension', dest='dec_hidden_dim', type=int, default=200)

    parser.add_argument('--optimizer', dest='optim', default='sgd')
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--clipvalue', type=float)
    parser.add_argument('--clipnorm', type=float)

    parser.add_argument('--lr_scheduler', default='constant')
    parser.add_argument('--lr_decay', type=float, default=1e-1)
    parser.add_argument('--lr_steps', nargs='+', type=float, default=[0.1, 0.5, 0.75, 0.9])
    parser.add_argument('--min_lr', type=float, default=1e-5)

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
        'gpu_id': args.gpu_id,
    }

def get_model(model_params):
    if model_params['model'] == 'naive':
        return NaiveSeq2Seq(
            model_params['encoder_vocab_count'],
            model_params['decoder_vocab_count'],
            model_params['emb_dim'],
            model_params['enc_hidden_dim'],
            model_params['dec_hidden_dim'],
            model_params['gpu_id'],
        )
    if model_params['model'] == 'global_attention':
        return Seq2SeqWithGlobalAttention(
            model_params['encoder_vocab_count'],
            model_params['decoder_vocab_count'],
            model_params['emb_dim'],
            model_params['enc_hidden_dim'],
            model_params['dec_hidden_dim'],
            model_params['gpu_id'],
        )
    if model_params['model'] == 'transformer':
        return TransformerEncoderDecoder(
            encoder_vocab_count=model_params['encoder_vocab_count'],
            decoder_vocab_count=model_params['decoder_vocab_count'],
            emb_dim=model_params['emb_dim'],
            encoder_hidden_dim=model_params['enc_hidden_dim'],
            decoder_hidden_dim=model_params['dec_hidden_dim'],
            head_count=4,
            feed_forward_hidden_dim=6,
            block_count=6,
        )

    raise ValueError('The model {} is not supported.'.format(model_params['model']))

def get_optimizer_params(args):
    optimizer_params = {}
    optimizer_params['type'] = args.optim

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

def get_lr_scheduler_params(args, train_data_loader):
    lr_scheduler_params = {}
    lr_scheduler_params['type'] = args.lr_scheduler
    lr_scheduler_params['kwargs'] = {}

    if args.lr_scheduler == 'constant':
        return lr_scheduler_params

    if args.lr_scheduler == 'multi_step':
        lr_scheduler_params['kwargs']['milestones'] = [
            int(args.epochs * step) for step in args.lr_steps
        ]
        lr_scheduler_params['kwargs']['gamma'] = args.lr_decay

        return lr_scheduler_params

    if args.lr_scheduler == 'cyclic':
        lr_scheduler_params['kwargs']['base_lr'] = args.min_lr
        lr_scheduler_params['kwargs']['max_lr'] = args.lr
        lr_scheduler_params['kwargs']['step_size_up'] = train_data_loader.iter_count
        lr_scheduler_params['kwargs']['mode'] = 'triangular'

        return lr_scheduler_params

    if args.lr_scheduler == 'cosine_annealing':
        lr_scheduler_params['kwargs']['T_max'] = train_data_loader.iter_count * 2
        lr_scheduler_params['kwargs']['eta_min'] = args.min_lr

        return lr_scheduler_params

    raise ValueError(
        'The learning rate scheduler {} is not supported.'.format(args.lr_scheduler))

def setup_output_dir(output_dir_path, args, model_params, optimizer_params):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))
    dump_json(model_params, os.path.join(output_dir_path, 'model.json'))
    dump_json(optimizer_params, os.path.join(output_dir_path, 'optimizer.json'))

def train_model(model, train_data_loader, optimizer, lr_scheduler, metric_manager, epoch):
    model.train()
    device = model.device

    train_loss_sum = 0.0
    train_data_count = 0
    with tqdm(total=len(train_data_loader), desc='Train') as pbar:
        for mb_inputs, mb_outputs in train_data_loader:
            mb_count = mb_inputs.shape[0]

            try:
                mb_inputs = torch.LongTensor(mb_inputs.transpose(1, 0)).to(device)
                mb_outputs = torch.LongTensor(mb_outputs.transpose(1, 0)).to(device)

                model.zero_grad()
                mb_probs = model(mb_inputs, mb_outputs[:-1])
                mb_train_loss = decoder_loss(mb_outputs[1:], mb_probs)
                mb_train_loss.backward()
                optimizer.step()

                mb_train_loss = mb_train_loss.cpu().data.numpy()
                train_loss_sum += mb_train_loss * mb_count
                train_data_count += mb_count

            except RuntimeError as error:
                logger.error(str(error))
                mb_train_loss = np.nan

            finally:
                torch.cuda.empty_cache()

            pbar.update(mb_count)
            pbar.set_postfix(OrderedDict(
                loss=mb_train_loss,
                plex=np.exp(mb_train_loss),
            ))

        lr_scheduler.step()
        train_loss = train_loss_sum / train_data_count
        logger.info('Train Loss: %f', train_loss)
        metric_manager.register_loss(train_loss, epoch, 'train')

def evaluate_model(model, valid_data_loader, metric_manager, epoch):
    model.eval()
    device = model.device

    valid_loss_sum = 0.0
    valid_data_count = 0
    with tqdm(total=len(valid_data_loader), desc='Valid') as pbar:
        for mb_inputs, mb_outputs in valid_data_loader:
            mb_count = mb_inputs.shape[0]

            try:
                mb_inputs = torch.LongTensor(mb_inputs.transpose(1, 0)).to(device)
                mb_outputs = torch.LongTensor(mb_outputs.transpose(1, 0)).to(device)

                mb_probs = model(mb_inputs, mb_outputs[:-1])
                mb_valid_loss = decoder_loss(mb_outputs[1:], mb_probs).cpu().data.numpy()

                valid_loss_sum += mb_valid_loss * mb_count
                valid_data_count += mb_count

            except RuntimeError as error:
                logger.error(str(error))
                mb_valid_loss = np.nan

            finally:
                torch.cuda.empty_cache()

            pbar.update(mb_count)
            pbar.set_postfix(OrderedDict(
                loss=mb_valid_loss,
                plex=np.exp(mb_valid_loss),
            ))

        valid_loss = valid_loss_sum / valid_data_count
        logger.info('Valid Loss: %f', valid_loss)
        metric_manager.register_loss(valid_loss, epoch, 'valid')

    return valid_loss

def train_loop(
        train_data_loader,
        valid_data_loader,
        model,
        optimizer,
        lr_scheduler,
        epochs,
        output_dir_path,
        model_name_format,
        best_monitored_metric=None,
    ):
    # Train Model
    metric_manager = MetricManager(output_dir_path, epochs)
    for epoch in range(epochs):
        logger.info('Start Epoch %s', epoch + 1)

        train_model(model, train_data_loader, optimizer, lr_scheduler, metric_manager, epoch)
        valid_loss = evaluate_model(model, valid_data_loader, metric_manager, epoch)

        # Save
        metric_manager.plot_loss('Loss', os.path.join(output_dir_path, 'loss.png'))
        metric_manager.save_score()

        monitored_metric = - valid_loss
        if best_monitored_metric is None or best_monitored_metric < monitored_metric:
            best_monitored_metric = monitored_metric
            logger.info('The current score is best.')
            if model_name_format:
                model_name = model_name_format.format(epoch=epoch + 1)
                logger.info('Save the model as %s', model_name)
                torch.save(model.state_dict(), os.path.join(output_dir_path, model_name))

    return best_monitored_metric

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    train_data_set = DataSet(is_training=True)
    train_data_set.input_data(args.train_data)
    train_data_loader = DataLoader(train_data_set, args.mb_size)
    preprocessor = train_data_set.preprocessor

    valid_data_set = DataSet(is_training=False, preprocessor=preprocessor)
    valid_data_set.input_data(args.valid_data)
    valid_data_loader = DataLoader(valid_data_set, args.mb_size)

    model_params = get_model_params(args)
    optimizer_params = get_optimizer_params(args)
    lr_scheduler_params = get_lr_scheduler_params(args, train_data_loader)
    if args.lang == 'ja_to_en':
        model_params['encoder_vocab_count'] = train_data_set.ja_vocab_count
        model_params['decoder_vocab_count'] = train_data_set.en_vocab_count
    elif args.lang == 'en_to_ja':
        model_params['encoder_vocab_count'] = train_data_set.en_vocab_count
        model_params['decoder_vocab_count'] = train_data_set.ja_vocab_count

    output_dir_path = args.output_dir_format.format(date=get_date_str())
    setup_output_dir(output_dir_path, dict(args._get_kwargs()), model_params, optimizer_params) #pylint: disable=protected-access
    preprocessor.save(os.path.join(output_dir_path, args.preprocessor))

    # Set up Model and Optimizer
    model = get_model(model_params)
    optimizer = get_torch_optimizer(model.parameters(), optimizer_params)
    lr_scheduler = get_torch_lr_scheduler(optimizer, lr_scheduler_params)

    train_loop(
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        output_dir_path=output_dir_path,
        model_name_format=args.model_name_format,
    )

if __name__ == '__main__':
    run()
