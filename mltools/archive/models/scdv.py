"""
SCDV モデルを定義しています。
"""
import numpy as np
from sklearn.mixture import GaussianMixture

from nlp_model.constant import PADDING
from nlp_model.utils import pad_texts
from nlp_model.interfaces import TextEmbedder

class SCDV(TextEmbedder):
    """
    Define a SCDV model.
    """
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.word_topic_vectors = None

    def fit(self, word_emb, n_components, max_iter):
        """
        Context-free な単語の分散表現を学習します。
        """
        clustering_model = GaussianMixture(n_components=n_components, max_iter=max_iter)
        clustering_model.fit(word_emb)
        cluster_proba = clustering_model.predict_proba(word_emb)

        word_emb_dim = word_emb.shape[1]
        self.word_topic_vectors = np.zeros((word_emb.shape[0], word_emb_dim * n_components))
        for i in range(n_components):
            self.word_topic_vectors[:, word_emb_dim * i: word_emb_dim * (i + 1)] = \
                cluster_proba[:, i].reshape(-1, 1) * word_emb

        return self

    @property
    def vocab_count(self):
        if self.word_topic_vectors is None:
            return -1
        return self.word_topic_vectors.shape[0]

    @property
    def emb_dim(self):
        if self.word_topic_vectors is None:
            return -1
        return self.word_topic_vectors.shape[1]

    @property
    def pad_index(self):
        return self.text_processor.token_dict.token2id[PADDING]

    def tokenize(self, text):
        return self.text_processor.tokenize(text)

    def index_tokens(self, tokens):
        return self.text_processor.index_tokens(tokens)

    def embed(self, texts):
        padded_texts, masks, _ = pad_texts(texts, self.pad_index)
        embeds = self.word_topic_vectors[padded_texts] * np.expand_dims(1 - masks, 2)

        return embeds

    def before_serialize(self):
        self.text_processor.before_serialize()

    def after_deserialize(self):
        self.text_processor.after_deserialize()
