"""
Pretrain a GPT model.
"""
import argparse
import logging
import os
import sqlite3
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from mltools.utils import set_seed, set_logging_handler, setup_output_dir#, get_optimizer
from mltools.models.gpt import GPT, GPTDataSet, GPTDataLoader
from mltools.preprocess.text_processor import TextProcessor

logger = logging.getLogger('Pretrain GPT')

parser = argparse.ArgumentParser()

parser.add_argument('--use_cuda', dest='use_cuda', help='Use CUDA or not', action='store_true')

parser.add_argument('--db_path', dest='db_path', help='database path to connect')
parser.add_argument('--db_column', dest='db_column', help='database column for input texts')
parser.add_argument('--cache_data_path')
parser.add_argument('--cache_text_preprocessor_path')
parser.add_argument(
    '--avoid_cache', dest='avoid_cache', help='Use cached data', action='store_true')
parser.add_argument('--output_dir', dest='output_dir')

parser.add_argument(
    '--emb_dim', dest='emb_dim', type=int,
    help='Dimension of the lookup layer',
    default=256
)
parser.add_argument(
    '--feedforward_hidden_dim', dest='feedforward_hidden_dim', type=int,
    help='Dimension of hidden layers in feed-forward network in transformers',
    default=128
)
parser.add_argument(
    '--head_count', dest='head_count', type=int,
    help='Count of heads in multi-head attentions in transformers',
    default=4
)
parser.add_argument(
    '--stack_count', dest='stack_count', type=int, help='Count of transformers', default=6
)

parser.add_argument('--optimizer', dest='optimizer', help='Optimizer', default='adabound')
parser.add_argument(
    '--lr', dest='lr', help='Learning rate of the optimizer', type=float, default=1e-4)
parser.add_argument(
    '--final_lr', dest='final_lr', help='Final learning rate of the optimizer',
    type=float, default=1e-1)
parser.add_argument(
    '--weight_decay', dest='weight_decay', help='weight decay of the optimizer',
    type=float, default=0)
parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=10.0)
parser.add_argument(
    '--lr_trigger', dest='lr_trigger', nargs='+',
    help='The finished ratio at which to exponentially shift a learning rate',
    type=int, default=[2, 128, 192, 232])

parser.add_argument('--mb_size', dest='mb_size', type=int, default=24)
parser.add_argument('--epochs', dest='epochs', type=int, default=256)
parser.add_argument('--seed', dest='seed', help='random seed to use this program')

args = parser.parse_args()

def get_model_params():
    return {
        "embed_params": {
            "args": {
                "emb_dim": args.emb_dim
            },
            "embed_type": "embedding"
        },
        "transformer_params": {
            "feedforward_hidden_dim": args.feedforward_hidden_dim,
            "head_count": args.head_count,
            "stack_count": args.stack_count
        }
    }

def get_optimizer_params():
    # if args.opt == 'sgd':
    #     return {
    #         "lr": args.lr,
    #         "decay": args.lr * args.weight_decay
    #     }

    # if args.opt == 'momentumsgd':
    #     return {
    #         "lr": args.lr,
    #         "decay": args.lr * args.weight_decay,
    #         "momentum": args.momentum
    #     }

    # if args.opt == 'adadelta':
    #     return {
    #         "decay": args.weight_decay,
    #     }

    if args.optimizer == 'adam':
        return {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }

    if args.optimizer == 'adabound':
        return {
            'lr': args.lr,
            'final_lr': args.final_lr,
            'weight_decay': args.weight_decay,
        }

    raise ValueError('The optimizer {} is not supported.'.format(args.optimizer))

def get_texts_from_db():
    """
    Get texts from a database.
    """
    sql = \
    """
    SELECT \"{column}\" FROM analysis_request where \"{column}\" <> \"\";
    """.format(column=args.db_column)    

    with sqlite3.connect(args.db_path) as conn:
        texts = pd.DataFrame(pd.read_sql_query(sql, conn))[args.db_column]

    return texts

def generate_data_set(processor_params):
    if not args.avoid_cache and \
            os.path.exists(args.cache_data_path) and \
            os.path.exists(args.cache_text_preprocessor_path):

        logger.info('Load data for GPT.')
        data_set = GPTDataSet.load(args.cache_data_path)
        text_preprocessor = TextProcessor.load(args.cache_text_preprocessor_path)

    else:
        logger.info('Start preprocessing data for GPT.')

        text_preprocessor = TextProcessor(**processor_params)

        texts = get_texts_from_db()
        data_set = GPTDataSet(texts, text_preprocessor)
        data_set.save(args.cache_data_path)
        text_preprocessor.save(args.cache_text_preprocessor_path)

        logger.info('Finish preprocessing data.')

    return data_set, text_preprocessor

def get_config_to_save(model_params, optimizer_params):
    return {
        'data_info': {
            'db_path': args.db_path,
            'db_column': args.db_column,
            'cache_data_path': args.cache_data_path,
            'cache_text_preprocessor_path': args.cache_text_preprocessor_path,
        },
        'model_type': args.model,
        'model_params': model_params,
        'optimizer_type': args.optimizer,
        'optimizer_params': optimizer_params,
        'mb_size': args.mb_size,
        'epochs': args.epochs,
        'seed': args.seed,
        'use_cuda': args.use_cuda
    }

def pretrain_gpt(data_loader, model, optimizer, epochs, output_dir_path):
    """
    Pretrain a GPT model.
    """
    data_count = len(data_loader)

    best_score = None
    scores = []
    for epoch in range(epochs):
        # Pretrain
        total_train_loss = 0.0
        total_count = 0
        logger.info('Start Epoch %s', epoch + 1)
        with tqdm(total=data_count, desc='Train') as pbar:
            for data in data_loader:
                mb_count = data[0].shape[1]

                try:
                    model.zero_grad()
                    loss = model.pretrain(data)
                    loss.backward()
                    optimizer.step()

                    mb_average_loss = loss.data.numpy()
                    total_train_loss += mb_average_loss * mb_count
                    total_count += mb_count

                except RuntimeError as _:
                    logger.error(str(_))
                    mb_average_loss = np.nan

                finally:
                    torch.cuda.empty_cache()

                pbar.update(mb_count)
                pbar.set_postfix(Loss='{:.4f}'.format(mb_average_loss))

        train_loss = total_train_loss / total_count
        logger.info('Train Loss: {:.4f}'.format(train_loss))

        # Set a score so that a better model will get a higher score.
        curr_score = - train_loss
        scores.append(curr_score)
        if best_score is None or best_score < curr_score:
            logger.info('The current epoch score is best for the current parameter tuning.')
            best_score = curr_score
            model.save(os.path.join(output_dir_path, 'pretrained_gpt.bin'))

        logger.info('End Epoch %s', epoch + 1)

def run():
    """
    Pretrain a GPT model.
    """
    processor_params = {
        "split_sentence": False,
        "text_type": "analysis_request",
        "tokenizer_params": {
            "model_path": "data/model/sentencepiece.model"
        },
        "tokenizer_type": "sentencepiece"
    }

    set_seed(args.seed)
    set_logging_handler([logger])

    data_set, text_preprocessor = generate_data_set(
        processor_params=processor_params)
    data_loader = GPTDataLoader(data_set, args.mb_size, args.use_cuda)

    logger.info('Start training a GPT model.')

    model_params = get_model_params()
    model = GPT(
        text_preprocessor,
        embed_params=model_params['embed_params'],
        transformer_params=model_params['transformer_params']
    )
    if args.use_cuda:
        model.cuda()
    optimizer_params = get_optimizer_params()
    optimizer = get_optimizer(model, args.optimizer, optimizer_params)
    output_dir_path = setup_output_dir(args.output_dir, dict(args._get_kwargs())) # pylint: disable=protected-access
    text_preprocessor.save(os.path.join(output_dir_path, 'preprocessor.bin'))

    pretrain_gpt(data_loader, model, optimizer, args.epochs, output_dir_path)

    logger.info('Finish training the model.')

if __name__ == '__main__':
    run()
