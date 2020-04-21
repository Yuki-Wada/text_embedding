"""
Inference a Japanese text into an English text using an Encoder-Decoder model.
"""
import os
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from mltools.utils import set_tensorflow_seed, set_logger, dump_json, get_date_str
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

    parser.add_argument('--embedding_dimension', dest='emb_dim', type=int, default=400)
    parser.add_argument('--encoder_hidden_dimension', dest='enc_hidden_dim', type=int, default=200)
    parser.add_argument('--decoder_hidden_dimension', dest='dec_hidden_dim', type=int, default=200)

    parser.add_argument('--optimizer', dest='optim', default='sgd')
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
        'emb_dim': args.emb_dim,
        'enc_hidden_dim': args.enc_hidden_dim,
        'dec_hidden_dim': args.dec_hidden_dim,
    }

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

def get_decoder_outputs(
        encoder_inputs,
        decoder_inputs,
        encoder_vocab_count,
        decoder_vocab_count,
        emb_dim,
        enc_hidden_dim,
        dec_hidden_dim,
    ):
    encoder_embedded = layers.Embedding(
        encoder_vocab_count,
        emb_dim,
        mask_zero=True,
    )(encoder_inputs)
    _, encoder_state_h, encoder_state_c = layers.LSTM(
        units=enc_hidden_dim,
        return_sequences=False,
        return_state=True,
    )(encoder_embedded)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder_embedded = layers.Embedding(decoder_vocab_count, emb_dim)(decoder_inputs)
    decoder_outputs, _, _ = layers.LSTM(
        units=dec_hidden_dim,
        return_sequences=True,
        return_state=True,
    )(
        decoder_embedded,
        initial_state=encoder_states,
    )
    decoder_outputs = layers.Dense(decoder_vocab_count)(decoder_outputs)

    return decoder_outputs

def plot_metrics(metrics, label, figure_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(metrics)) + 1, metrics)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)

    x_ax = ax.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(figure_path)

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

    # Set up Model
    encoder_inputs = layers.Input(shape=(None,))
    decoder_inputs = layers.Input(shape=(None,))
    decoder_outputs = get_decoder_outputs(
        encoder_inputs,
        decoder_inputs,
        train_data_loader.ja_vocab_count,
        train_data_loader.en_vocab_count,
        model_params['emb_dim'],
        model_params['enc_hidden_dim'],
        model_params['dec_hidden_dim'],
    )

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.summary()

    # Set up Optimizer
    optimizer, _ = get_keras_optimizer(optimizer_params)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
        ]
    )

    # Train Model
    train_losses = []
    valid_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(epochs):
        logger.info('Start Epoch %s', epoch + 1)

        # Train
        train_loss_sum = 0.0
        train_accuracy_sum = 0.0
        train_data_count = 0
        with tqdm(total=len(train_data_loader), desc='Train') as pbar:
            for mb_inputs, mb_outputs in train_data_loader:
                mb_count = mb_inputs.shape[0]
                mb_train_loss, mb_train_accuracy = model.train_on_batch(
                    [mb_inputs, mb_outputs[:, :-1]],
                    mb_outputs[:, 1:],
                )

                train_loss_sum += mb_train_loss * mb_count
                train_accuracy_sum += mb_train_accuracy * mb_count
                train_data_count += mb_count

                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=train_loss_sum / train_data_count,
                    accu=train_accuracy_sum / train_data_count,
                ))

        # Valid
        valid_loss_sum = 0.0
        valid_accuracy_sum = 0.0
        valid_data_count = 0
        with tqdm(total=len(valid_data_loader), desc='Valid') as pbar:
            for mb_inputs, mb_outputs in valid_data_loader:
                mb_count = mb_inputs.shape[0]

                mb_valid_loss, mb_valid_accuracy = model.test_on_batch(
                    [mb_inputs, mb_outputs[:, :-1]],
                    mb_outputs[:, 1:],
                )
                valid_loss_sum += mb_valid_loss * mb_count
                valid_accuracy_sum += mb_valid_accuracy * mb_count
                valid_data_count += mb_count

                pbar.update(mb_count)
                pbar.set_postfix(OrderedDict(
                    loss=valid_loss_sum / valid_data_count,
                    accu=valid_accuracy_sum / valid_data_count,
                ))

        valid_loss = valid_loss_sum / valid_data_count
        valid_accuracy = valid_accuracy_sum / valid_data_count
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Plot Metrics
        figure_path = os.path.join(output_dir_path, 'loss.png')
        plot_metrics(valid_losses, 'Loss', figure_path)
        figure_path = os.path.join(output_dir_path, 'accuracy.png')
        plot_metrics(valid_accuracies, 'Accuracy', figure_path)

        # Save Model.
        monitored_metric = valid_loss
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
