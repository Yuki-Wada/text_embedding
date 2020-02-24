"""
Train a sentencepiece tokenizer by inputting sentences written in filtered repair data.
"""
import os
import sentencepiece as spm

from nlp_model.utils import read_sjis_csv, read_json

def train_sentencepiece(texts, save_model_prefix):
    """
    Train a sentencepiece tokenizer.
    """
    text_file_path = 'data/feature/preprocess/sentences_for_training_sentencepiece.txt'
    texts.dropna().to_csv(text_file_path, encoding='utf8')

    spm_args = '--input={} --model_prefix={} --vocab_size=3000'.format(
        text_file_path, save_model_prefix)
    spm.SentencePieceTrainer.Train(spm_args)

    os.remove(text_file_path)

    spp = spm.SentencePieceProcessor()
    spp.Load('{}.model'.format(save_model_prefix))
