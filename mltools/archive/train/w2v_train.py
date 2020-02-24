"""
Train a Word2Vec model and obtain its word embedding.
"""
import logging
import dill
import numpy as np

from nlp_model.utils import set_random_seed
from nlp_model.preprocess.text_processor import TextProcessor
from nlp_model.models.w2v_wrapper import W2VWrapper

def get_word_emb_from_w2v(w2v_model, token_dict):
    """
    Obtain a word embedding from a Word2Vec model.
    """
    w2v_word_emb = w2v_model.wv.vectors
    words = w2v_model.wv.index2word

    word_index = [token_dict.token2id[v] for v in words]
    mask = np.zeros((len(token_dict),))
    mask[word_index] = 1

    emb_dim = w2v_word_emb.shape[1]
    word_emb = np.zeros((len(token_dict), emb_dim))
    word_emb[mask == 1] = w2v_word_emb
    word_emb[mask == 0] = np.random.rand(np.sum(np.logical_not(mask)), emb_dim)

    return word_emb

def w2v_train(
        texts, save_model_path, save_word_emb_path,
        processor_params, w2v_params, epochs, seed=None):
    """
    Train a Word2Vec model and obtain its word embedding.
    """
    logging.info('Start training a Word2Vec Model.')

    # set NumPy random seed
    set_random_seed(seed)

    text_processor = TextProcessor(**processor_params)

    texts = texts.apply(text_processor.tokenize)
    texts = texts.apply(text_processor.add_bos_and_eos)
    text_processor.add_documents(texts)

    w2v_model = W2VWrapper(text_processor, w2v_params)
    w2v_model.build_vocab(texts)
    w2v_model.train(texts, total_examples=len(texts), epochs=epochs)

    w2v_model.save(save_model_path)
    word_emb = w2v_model.get_word_emb()
    with open(save_word_emb_path, 'wb') as _:
        dill.dump(word_emb, _)

    logging.info('Finish training.')
