"""
Train a Poincare Embedding model and obtain its word embedding.
"""
import numpy as np
from gensim.models.poincare import PoincareModel

from nlp_model.constant import EPSILON
from nlp_model.utils import pad_texts
from nlp_model.interfaces import TextEmbedder

def dist_on_poincare_disk(u, v):
    delta = 2 * np.sum((u - v) * (u - v)) / (1 - np.sum(u * u)) / (1 - np.sum(v * v))
    arcosh = lambda x: np.log(x + np.sqrt((x + 1) * (x - 1)))
    return arcosh(1 + delta)

def poincare_to_klein(u, emb_axis=None):
    dist = np.sqrt(np.sum(u * u, axis=emb_axis, keepdims=True))
    return 2 * dist / (1 + dist * dist) * u / dist

def klein_to_poincate(u, emb_axis=None):
    dist = np.sqrt(np.sum(u * u, axis=emb_axis, keepdims=True))
    return (1 - np.sqrt(1 - dist * dist)) * u / (dist + EPSILON) / (dist + EPSILON)

def discount(u, emb_axis=None):
    return 1 / np.sqrt(1 - np.sum(u * u, axis=emb_axis))

def add_on_poincare_disk(u, v):
    numerator = (1 + 2 * np.sum(u * v) + np.sum(v * v)) * u + (1 - np.sum(u * u)) * v
    denominator = 1 + 2 * np.sum(u * v) + np.sum(u * u) * np.sum(v * v)
    return numerator / denominator

def add_on_klein_disk(u, v):
    gamma = discount(u)
    numerator = u + v + gamma / (1 + gamma) * (np.sum(u * v) * u - np.sum(u * u) * v)
    denominator = 1 + np.sum(u * v)
    return numerator / denominator

def scalar(r, u):
    numerator = (np.power(1 + np.sqrt(np.sum(u * u)), r) - np.power(1 - np.sqrt(np.sum(u * u)), r)) * u
    denominator = (np.power(1 + np.sqrt(np.sum(u * u)), r) + np.power(1 - np.sqrt(np.sum(u * u)), r)) * np.sqrt(np.sum(u * u))
    return numerator / denominator

def gyr_on_poincare_disk(a, b):
    left = lambda c: add_on_poincare_disk(a, add_on_poincare_disk(b, c))
    minus_a_plus_b = - add_on_poincare_disk(a, b)
    return lambda c: add_on_poincare_disk(minus_a_plus_b, left(c))

def gyr_on_klein_disk(a, b):
    left = lambda c: add_on_klein_disk(a, add_on_klein_disk(b, c))
    minus_a_plus_b = - add_on_klein_disk(a, b)
    return lambda c: add_on_klein_disk(minus_a_plus_b, left(c))

def weighted_sum_on_klein_disk(u, weights, seq_axis=1, emb_axis=2, keepdims=False):
    discounted_weights = weights * discount(u, emb_axis=emb_axis)
    discounted_weights /= np.sum(discounted_weights, axis=seq_axis, keepdims=True) + EPSILON
    return np.sum(np.expand_dims(discounted_weights, emb_axis) * u, axis=seq_axis, keepdims=keepdims)

def weighted_sum_on_poincare_disk(u, weights, seq_axis=1, emb_axis=2):
    klein_u = poincare_to_klein(u, emb_axis=emb_axis)
    weighted_klein_u = weighted_sum_on_klein_disk(klein_u, weights, seq_axis=seq_axis, emb_axis=emb_axis, keepdims=True)
    return np.squeeze(klein_to_poincate(weighted_klein_u, emb_axis=emb_axis), axis=seq_axis)

class PoincareEmbedding(TextEmbedder):
    """
    Define a Poincare Embedding model.
    """
    def __init__(self, text_processor, texts, window, poincare_model):
        self.text_processor = text_processor

        relations = []
        for text in texts:
            for i, word in enumerate(text):
                curr_window = np.random.randint(window + 1)
                neighbors = text[max(0, i - window): i]
                relations += [(word, neighbor) for neighbor in neighbors]
                neighbors = text[i + 1: min(len(text), i + curr_window)]
                relations += [(word, neighbor) for neighbor in neighbors]

        self.internal_model = PoincareModel(relations, **poincare_model)

    @property
    def vocab_count(self):
        return self.internal_model.kv.vectors.shape[0]

    @property
    def emb_dim(self):
        return self.internal_model.kv.vectors.shape[1]

    @property
    def pad_index(self):
        return self.text_processor.pad_index

    def tokenize(self, text):
        return self.text_processor.tokenize(text)

    def index_tokens(self, tokens):
        return [
            self.internal_model.kv.vocab[token].index for token in tokens
            if token in self.internal_model.kv.vocab]

    def train(self, train_params):
        """
        Train this model.
        """
        self.internal_model.train(**train_params)

    def embed(self, texts):
        padded_texts, _, _ = pad_texts(texts)
        return self.internal_model.kv.vectors[padded_texts]

    def convert(self, embeds, weights=None):
        return weighted_sum_on_poincare_disk(embeds, weights)

    def before_serialize(self):
        self.text_processor.before_serialize()

    def after_deserialize(self):
        self.text_processor.after_deserialize()
