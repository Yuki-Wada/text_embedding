"""
文章をトークン化するためのクラス
"""
import re
import dill

import numpy as np
from gensim.corpora import Dictionary
import sentencepiece as spm
import MeCab

from nlp_model.utils import add_bos_and_eos

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
CLS = '<CLS>'
SEP = '<SEP>'
MASK = '<MASK>'

def get_mecab_tokenizer():
    return MeCab.Tagger()

class TextProcessor:
    """
    Define the class which process texts.
    """
    def __init__(self):
        self.special_token_dict = Dictionary([[BOS, EOS, UNK, PAD, CLS, SEP, MASK]])
        self.vocab_dict = Dictionary()

    def __len__(self):
        return len(self.special_token_dict) + len(self.vocab_dict)

    def get_tokenization_method(self, tokenizer):
        def tokenize(sentence):
            return tokenizer.tokenize(sentence)
        return tokenize

    def add_documents(self, texts):
        self.vocab_dict.add_documents(texts)

    def get_tf(self, texts):
        table = np.zeros((len(texts), len(self)))
        for i, text in enumerate(texts):
            for word, freq in self.doc2bow(text):
                table[i, word] += freq
        return table

    def get_idf(self):
        counts = np.array([count for _, count in sorted(
            self.vocab_dict.dfs.items(), key=lambda x: int(x[0]))])
        idf = np.log(self.vocab_dict.num_docs / (counts + 1))

        return idf

    def doc2bow(self, text):
        return self.vocab_dict.doc2bow(text)

    def get_special_token(self, token):
        if token in self.special_token_dict.token2id:
            return self.special_token_dict.token2id[token]
        raise ValueError('{} is not in the special token dict.'.format(token))

    @property
    def bos_index(self):
        return self.get_special_token(BOS)

    @property
    def eos_index(self):
        return self.get_special_token(EOS)

    @property
    def pad_index(self):
        return self.get_special_token(PAD)

    @property
    def unk_index(self):
        """
        Return a UNK token's index used in this model.
        """
        return self.get_special_token(UNK)

    @property
    def cls_index(self):
        """
        Return a CLS token's index used in this model.
        """
        return self.get_special_token(CLS)

    @property
    def sep_index(self):
        """
        Return a SEP token's index used in this model.
        """
        return self.get_special_token(SEP)

    @property
    def mask_index(self):
        """
        Return a MASK token's index used in this model.
        """
        return self.get_special_token(MASK)

    def add_bos_and_eos(self, text):
        """
        Add the BOS token to the begin of the input text and the EOS token to the end of it.
        """
        return [BOS] + text + [EOS]

    def index_tokens(self, tokens):
        return self.vocab_dict.doc2idx(tokens, unknown_word_index=self.unk_index)

    def save(self, save_path):
        """
        Serialize this object.
        """
        with open(save_path, 'wb') as _:
            dill.dump(self, _)

    @classmethod
    def load(cls, load_path):
        """
        Deserialize an object from a file.
        """
        with open(load_path, 'rb') as _:
            model = dill.load(_)
        assert isinstance(model, cls), \
            'Input appropriate {} model file path.'.format(cls.__name__)
        return model
