"""
Define an original Word2Vec model using skipgram and negative sampling.
"""
from typing import List, Union
import numpy as np
from gensim.corpora import Dictionary

from mltools.model.word2vec_impl import get_sg_ns_grad #pylint: disable=no-name-in-module

class MyWord2Vec:
    def __init__(
            self,
            dictionary: Dictionary,
            window: int = 5,
            size: int = 100,
            negative: int = 5,
            ns_exponent: float = 0.75,
            alpha: float = 0.025,
            workers: int = 4):
        self.window = window
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.lr = np.float32(alpha)
        self.workers = workers

        self._dictionary = dictionary
        self._size = size

        self._w_in = np.random.randn(self._size, self.vocab_count).astype(np.float32)
        self._w_out = np.random.randn(self.vocab_count, self._size).astype(np.float32)

    @property
    def vocab_count(self) -> int:
        return len(self._dictionary)

    @property
    def vocab_ns_prob(self) -> List[float]:
        if not hasattr(self, '_vocab_ns_prob'):
            freq = np.array([self._dictionary.dfs[i] for i in range(self.vocab_count)])
            vocab_ns_prob = (freq ** self.ns_exponent) / \
                (np.sum(freq ** self.ns_exponent) + 1e-18)
            self._vocab_ns_prob = vocab_ns_prob.tolist() #pylint: disable=attribute-defined-outside-init
        return self._vocab_ns_prob

    def train(self, texts: List[List[int]]):
        get_sg_ns_grad(
            texts, self.window, self.negative, self.vocab_ns_prob, self._w_in, self._w_out, self.lr)

    def most_similar(
            self,
            positive: Union[str, List[str], None] = None,
            negative: Union[str, List[str], None] = None,
            topn: int = 10):

        vector = np.zeros((self._size,)).astype(np.float32)
        if positive:
            if isinstance(positive, str):
                vector += self._w_in[:, self._dictionary.token2id[positive]]
            elif isinstance(positive, list):
                for token in positive:
                    vector += self._w_in[:, self._dictionary.token2id[token]]
        if negative:
            if isinstance(negative, str):
                vector += self._w_in[:, self._dictionary.token2id[negative]]
            elif isinstance(negative, list):
                for token in negative:
                    vector -= self._w_in[:, self._dictionary.token2id[token]]

        cos_sim = np.matmul(vector, self._w_in) / np.linalg.norm(vector, ord=2) / \
            np.linalg.norm(self._w_in, ord=2, axis=0)

        top_words = []
        for index in np.argsort(cos_sim)[::-1]:
            word = self._dictionary[index]
            top_words.append(word)
            if len(top_words) >= topn:
                break

        return top_words
