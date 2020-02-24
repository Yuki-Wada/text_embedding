"""
Convert an input text to a vector by using a model with a word embedding.
"""
import logging
import dill
import numpy as np

import torch

from nlp_model.constant import EPSILON
from nlp_model.utils import pad_texts

from nlp_model.models.scdv import SCDV
from nlp_model.models.elmo import ELMo
from nlp_model.models.gpt import GPT
from nlp_model.models.bert import BERT
from nlp_model.models.poincare_embedding import PoincareEmbedding

def load_model(model_type, load_model_path):
    """
    Load the model corresponding to a model type.
    """
    if model_type == 'scdv':
        return SCDV.load(load_model_path)
    if model_type == 'elmo':
        return ELMo.load(load_model_path)
    if model_type == 'gpt':
        return GPT.load(load_model_path)
    if model_type == 'bert':
        return BERT.load(load_model_path)
    if model_type == 'poincare_embedding':
        return PoincareEmbedding.load(load_model_path)
    raise ValueError('You cannot select the model type of {}.'.format(model_type))

def convert(indexed_texts, model, weight, mb_size):
    text_vectors = np.zeros((len(indexed_texts), model.emb_dim))
    for mb_index_begin in range(0, len(indexed_texts), mb_size):
        mb_indexed_texts = indexed_texts[mb_index_begin:mb_index_begin+mb_size]
        mb_texts, mb_masks, _ = pad_texts(mb_indexed_texts, model.pad_index)

        curr_weights = weight[mb_texts] * (1 - mb_masks)
        curr_weights /= np.sum(curr_weights, axis=1, keepdims=True) + EPSILON

        embed = model.embed(mb_indexed_texts)
        if isinstance(embed, torch.Tensor):
            embed = embed.data.numpy()

        text_vectors[mb_index_begin:mb_index_begin+mb_size] = \
            model.convert(embed, curr_weights)

    return text_vectors

def batch_convert(model, texts, keys, save_vector_path, mb_size=32):
    logging.info(
        'Start converting an input text to a vector by {}.'.format(model.__class__.__name__))

    idf = model.text_processor.get_idf()
    texts = texts.apply(model.tokenize)
    texts = texts.apply(model.index_tokens)

    text_vectors = convert(texts, model, idf, mb_size)

    with open(save_vector_path, 'wb') as _:
        dill.dump((keys, text_vectors), _)

    logging.info('Finish converting.')
