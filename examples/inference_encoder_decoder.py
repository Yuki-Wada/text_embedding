"""
Inference a Japanese text into an English text using an Encoder-Decoder model.
"""
import os
import argparse
import logging
import json
from tqdm import tqdm

from mltools.utils import set_tensorflow_seed, set_logger, dump_json
from mltools.model.encoder_decoder import NaiveSeq2Seq, \
    Seq2SeqWithGlobalAttention, TransformerEncoderDecoder
from mltools.dataset.japanese_english_bilingual_corpus \
    import BilingualPreprocessor as Preprocessor, BilingualDataSet as DataSet, \
        BilingualDataLoader as DataLoader

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inference_data', nargs='+', required=True)
    parser.add_argument('--model_path')
    parser.add_argument('--model_params')
    parser.add_argument('--preprocessor')
    parser.add_argument('--seq_len', type=int, default=100)

    parser.add_argument('--mb_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--seed', type=int, help='random seed for initialization')

    args = parser.parse_args()

    return args

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
    if model_params['model'] == 'transformer': 
        return TransformerEncoderDecoder(
            encoder_vocab_count=model_params['ja_vocab_count'],
            decoder_vocab_count=model_params['en_vocab_count'],
            emb_dim=model_params['emb_dim'],
            encoder_hidden_dim=model_params['enc_hidden_dim'],
            decoder_hidden_dim=model_params['dec_hidden_dim'],
            head_count=4,
            feed_forward_hidden_dim=6,
            block_count=6,
        )
    raise ValueError('The model {} is not supported.'.format(model_params['model']))

def setup_output_dir(output_dir_path, args, model_params, optimizer_params):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))
    dump_json(model_params, os.path.join(output_dir_path, 'model.json'))
    dump_json(optimizer_params, os.path.join(output_dir_path, 'optimizer.json'))

def inference_encoder_decoder(
        inference_data_loader,
        model_params,
        model_path,
        preprocessor,
        seq_len,
        seed=None,
    ):
    set_tensorflow_seed(seed)

    # Set up Model
    model = get_model(model_params)
    model.load_weights(model_path)

    # Inference
    inference_data_count = 0
    en_texts = []
    decoded_texts = []
    with tqdm(total=len(inference_data_loader), desc='Inference') as pbar:
        for mb_inputs, mb_outputs in inference_data_loader:
            mb_count = mb_inputs.shape[0]
            mb_decoded = model.inference(mb_inputs, preprocessor.en_begin_of_encode_index, seq_len)

            for indices in mb_outputs:
                en_texts.append(
                    ' '.join([
                        preprocessor.en_dictionary[index] for index in indices
                    ])
                )
            for indices in mb_decoded:
                decoded_texts.append(
                    ' '.join([
                        preprocessor.en_dictionary[index] for index in indices
                    ])
                )
            inference_data_count += mb_count
            pbar.update(mb_count)

def run():
    set_logger()
    args = get_args()
    set_tensorflow_seed(args.seed)

    with open(args.model_params, 'r') as f:
        model_params = json.load(f)

    preprocessor = Preprocessor.load(args.preprocessor)

    valid_data_set = DataSet(is_training=False, preprocessor=preprocessor)
    valid_data_set.input_data(args.inference_data)
    valid_data_loader = DataLoader(valid_data_set, args.mb_size)

    inference_encoder_decoder(
        inference_data_loader=valid_data_loader,
        model_params=model_params,
        model_path=args.model_path,
        preprocessor=preprocessor,
        seq_len=args.seq_len,
        seed=args.seed,
    )

if __name__ == '__main__':
    run()
