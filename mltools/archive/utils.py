import os
import json
import dill
import numpy as np
import pandas as pd
from jinja2 import Template

import torch
import torch.nn as nn
import torch.optim as optim
import adabound

def read_sjis_csv(path: str, usecols=None):
    """
    文字コードが Shift-JIS であり、ファイル名が日本語である csv ファイルを読み込みます
    (ファイル名が日本語であるファイルを読み込むためには、engine='python' を指定する必要があるが、
    このことをいつも忘れてしまうため、この関数を作成)。

    パラメータ
    ----------
    path: str

    返り値
    ----------
    df: pandas.DataFrame
        Shift-JIS の csv ファイルを Pandas の DataFrame 形式で読み込んだ結果が入っています。
    """
    csv_df = pd.read_csv(path, engine='python', encoding='cp932', usecols=usecols)
    return csv_df

def save_sjis_csv(data_frame: pd.DataFrame, path: str):
    """
    csv ファイルを Shift-JIS の文字コードで保存します。
    """
    data_frame.to_csv(path, encoding='cp932', index=False)

def read_json(path: str):
    with open(path, 'r') as _:
        config = json.load(_)
    return config

def read_json_template(path: str, **kwargs):
    with open(path, 'r') as _:
        config_template = _.read()
    config_str = Template(config_template).render(**kwargs)
    config = json.loads(config_str)
    return config

def get_cosine_sims(vec1: np.ndarray, vec2: np.ndarray):
    """
    vec1 と vec2 の cosine 類似度を計算します。

    パラメータ
    ----------
    vec1: numpy.ndarray, shape (count1, dim)
    vec2: numpy.ndarray, shape (count2, dim)

    返り値
    ----------
    cos_sims: numpy.ndarray, shape (count1, count2)
        cos_sims[i][j] には vec1[i] と vec2[j] の cosine 類似度の計算結果が入っています。
    """
    norm_vec1 = np.sqrt(np.sum(vec1 ** 2, axis=1, keepdims=True)) + EPSILON
    norm_vec2 = np.sqrt(np.sum(vec2 ** 2, axis=1, keepdims=True)) + EPSILON
    cos_sims = np.dot(vec1 / norm_vec1, (vec2 / norm_vec2).T)
    return cos_sims

def add_bos_and_eos(text):
    """
    Add the BOS token to the begin of the input text and the EOS token to the end of it.
    """
    return [BOS] + text + [EOS]

def pad_texts(texts, pad_index=0):
    """
    文章をパディングすることで、それぞれ長さの異なる文章群を 2 次元の numpy.ndarray の形にします。
    文章の長さ、パディング領域のマスクも同時に返します。
    """
    text_count = len(texts)
    lengths = np.array([len(text) for text in texts])
    max_length = np.max(lengths)

    masks = np.ones((text_count, max_length))
    padded_texts = np.ones((text_count, max_length)).astype(np.int) * pad_index

    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            padded_texts[i, j] = word
            masks[i, j] = 0

    return padded_texts, masks, lengths

def split_indices(data_count, ratios=np.array([9, 1]), do_shuffle=True):
    ratios = np.array(ratios).reshape(-1)
    accum_ratios = np.zeros(len(ratios) + 1)
    accum_ratios[1:] = ratios
    accum_ratios = np.cumsum(accum_ratios)
    accum_ratios = accum_ratios / accum_ratios[-1]

    indices = np.arange(data_count)
    if do_shuffle:
        indices = np.random.permutation(indices)
    split_pos = (accum_ratios * data_count).astype(np.int32)
    return [indices[split_pos[i] : split_pos[i + 1]] for i in range(len(split_pos)-1)]

def get_mb_indices(indices: np.array, mb_size: int, shuffle_type='no_shuffle'):
    """
    indices を mb_size の個数ずつ返すイテレータです。
    do_shuffle を True にすることで、indices をシャッフルした後に返します。
    """
    if shuffle_type == 'no_shuffle':
        indices = np.random.permutation(indices)
        for i in range(0, len(indices), mb_size):
            yield indices[i : i + mb_size]
    if shuffle_type == 'totally_shuffle':
        for i in range(0, len(indices), mb_size):
            yield indices[i : i + mb_size]        
    if shuffle_type == 'shuffle_begin_index':
        mb_count = np.ceil(len(indices) / mb_size).astype(np.int32)
        for i in np.random.permutation(np.arange(mb_count)):
            yield indices[i * mb_size : (i + 1) * mb_size]

def cross_entropy(outputs, labels):
    return np.sum([-np.log(outputs)[i, label] for i, label in enumerate(labels)])


def get_optimizer(model, optimizer_type, args):
    """
    Obtain an optimizer.
    """
    if optimizer_type == 'adam':
        # {
        #     "type": "adam",
        #     "args": {
        #         "lr": 1e-4,
        #         "weight_decay": 1e-5
        #     }
        # }
        return optim.Adam(model.parameters(), **args)
    elif optimizer_type == 'adabound':
        # {
        #     "type": "adabound",
        #     "args": {
        #         "lr": 1e-4,
        #         "final_lr": 1e-1,
        #         "weight_decay": 1e-5
        #     }
        # }
        return adabound.AdaBound(model.parameters(), **args)

def get_embed_layer(embed_type, args):
    if embed_type == 'embedding':
        # {
        #     "embed_type": "embedding",
        #     "args": {
        #         "vocab_count": 1000,
        #         "emb_dim": 250
        #     }
        # }
        emb_dim = args['emb_dim']
        layer = nn.Embedding(args['vocab_count'], args['emb_dim'])
        layer.embed = layer.forward
    elif embed_type == 'from_pretrained':
        # {
        #     "embed_type": "from_pretrained",
        #     "args": {
        #         "vocab_count": 1000,
        #         "word_emb_path": "data/model/w2v_word_emb.pkl"
        #     }
        # }
        with open(args['word_emb_path'], 'rb') as _:
            word_emb = dill.load(_)
        assert word_emb.shape[0] == args['vocab_count'], \
            'The loaded word embedding vocabraries must be equal to "vocab_count" in args.'
        emb_dim = word_emb.shape[1]
        layer = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        layer.embed = layer.forward
    elif embed_type == 'elmo':
        # {
        #     "embed_type": "elmo",
        #     "args": {
        #         "model_path": "data/model/elmo_model.pkl"
        #     }
        # }
        with open(args['model_path'], 'rb') as _:
            layer = dill.load(_)
        emb_dim = layer.emb_dim

    return layer, emb_dim

def use_cuda():
    return os.path.exists('configs/CUDA_ENABLED')