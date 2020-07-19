"""
Inference a Japanese text into an English text using an Encoder-Decoder model.
"""
import argparse
import logging
import json
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from mltools.utils import set_seed, set_logger
from mltools.model.encoder_decoder import calc_bleu_scores, NaiveSeq2Seq, \
    Seq2SeqWithGlobalAttention, TransformerEncoderDecoder
# from mltools.dataset.japanese_english_bilingual_corpus import \
#     BilingualPreprocessor as Preprocessor, BilingualDataSet as DataSet, \
#     BilingualDataLoader as DataLoader
from mltools.dataset.tanaka_corpus import \
    BilingualPreprocessor as Preprocessor, TanakaCorpusDataSet as DataSet, \
    TanakaCorpusDataLoader as DataLoader

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=-1)

    parser.add_argument('--inference_data', nargs='+', required=True)
    parser.add_argument('--lang', default='ja_to_en')
    parser.add_argument('--model_params')
    parser.add_argument('--model_weight')
    parser.add_argument('--preprocessor')

    parser.add_argument('--decoding_type', default='random_choice')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--breadth_len', type=int, default=8)

    parser.add_argument('--inference_csv')

    parser.add_argument('--mb_size', type=int, default=4, help='minibatch size')
    parser.add_argument('--seed', type=int, help='random seed for initialization')

    args = parser.parse_args()

    return args

def get_decoding_params(args, preprocessor):
    decoding_params = {
        'decoding_type': args.decoding_type,
        'seq_len': args.seq_len,
        'begin_of_encode_index': preprocessor.en_begin_of_encode_index,
    }
    if decoding_params['decoding_type'] == 'beam_search':
        decoding_params['breadth_len'] = args.breadth_len

    return decoding_params

def get_model(model_params):
    if model_params['model'] == 'naive':
        return NaiveSeq2Seq(
            model_params['encoder_vocab_count'],
            model_params['decoder_vocab_count'],
            model_params['emb_dim'],
            model_params['hidden_dim'],
            model_params['bidirectional'],
            model_params['gpu_id'],
        )
    if model_params['model'] == 'global_attention':
        return Seq2SeqWithGlobalAttention(
            model_params['encoder_vocab_count'],
            model_params['decoder_vocab_count'],
            model_params['emb_dim'],
            model_params['hidden_dim'],
            model_params['bidirectional'],
            model_params['gpu_id'],
        )
    if model_params['model'] == 'transformer':
        return TransformerEncoderDecoder(
            encoder_vocab_count=model_params['encoder_vocab_count'],
            decoder_vocab_count=model_params['decoder_vocab_count'],
            model_dim=model_params['model_dim'],
            head_count=model_params['head_count'],
            feed_forward_hidden_dim=model_params['feed_forward_hidden_dim'],
            encoder_block_count=model_params['encoder_block_count'],
            decoder_block_count=model_params['decoder_block_count'],
            gpu_id=model_params['gpu_id'],
        )

    raise ValueError('The model {} is not supported.'.format(model_params['model']))

def tokenize_indices(mb_indices, index_to_token, eos_index=-1):
    texts = []

    for indices in mb_indices:
        text = []
        for index in indices:
            if index == eos_index:
                break
            text.append(index_to_token[index])

        texts.append(' '.join(text))

    return texts

def inference_encoder_decoder(
        inference_data_loader,
        model,
        preprocessor,
        decoding_params,
        inference_csv_path,
    ):
    model.eval()
    device = model.device

    # Inference
    inference_data_count = 0
    input_texts = []
    actual_output_texts = []
    predicted_output_texts = []
    bleu2_scores = []
    bleu4_scores = []
    with tqdm(total=len(inference_data_loader), desc='Inference') as pbar:
        for mb_inputs, mb_outputs in inference_data_loader:
            mb_count = mb_inputs.shape[0]

            try:
                mb_input_tensor = torch.LongTensor(mb_inputs.transpose(1, 0)).to(device)

                _, mb_predicted = model.inference(mb_input_tensor, decoding_params)

                mb_outputs = mb_outputs[:, 1:]

                input_texts += tokenize_indices(
                    mb_inputs, preprocessor.en_dictionary, preprocessor.en_eos_index)
                actual_output_texts += tokenize_indices(
                    mb_outputs, preprocessor.ja_dictionary, preprocessor.ja_eos_index)
                predicted_output_texts += tokenize_indices(
                    mb_predicted, preprocessor.ja_dictionary, preprocessor.ja_eos_index)

                mb_bleu2_scores = calc_bleu_scores(
                    mb_outputs, mb_predicted, preprocessor.ja_eos_index, max_n=2)
                bleu2_scores += mb_bleu2_scores
                mb_bleu2_score = np.mean(mb_bleu2_scores)

                mb_bleu4_scores = calc_bleu_scores(
                    mb_outputs, mb_predicted, preprocessor.ja_eos_index, max_n=4)
                bleu4_scores += mb_bleu4_scores
                mb_bleu4_score = np.mean(mb_bleu4_scores)

            except RuntimeError as error:
                logger.error(str(error))

            finally:
                torch.cuda.empty_cache()

            inference_data_count += mb_count
            pbar.update(mb_count)
            pbar.set_postfix(OrderedDict(
                bleu2=mb_bleu2_score,
                bleu4=mb_bleu4_score,
            ))

            pd.DataFrame({
                'Japanese': input_texts,
                'English (Actual)': actual_output_texts,
                'English (Inference)': predicted_output_texts,
                'BLEU2 Score': bleu2_scores,
                'BLEU4 Score': bleu4_scores,
            }).to_csv(
                inference_csv_path, index=False,
            )

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    preprocessor = Preprocessor.load(args.preprocessor)

    valid_data_set = DataSet(is_training=False, preprocessor=preprocessor)
    valid_data_set.input_data(args.inference_data)
    valid_data_loader = DataLoader(valid_data_set, args.mb_size)

    # Set up Model
    with open(args.model_params, 'r') as f:
        model_params = json.load(f)
    model = get_model(model_params)
    model.load_state_dict(torch.load(args.model_weight))

    decoding_params = get_decoding_params(args, preprocessor)

    inference_encoder_decoder(
        inference_data_loader=valid_data_loader,
        model=model,
        preprocessor=preprocessor,
        decoding_params=decoding_params,
        inference_csv_path=args.inference_csv,
    )

if __name__ == '__main__':
    run()
