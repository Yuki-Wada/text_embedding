"""
Define an original Word2Vec model using skipgram and negative sampling.
"""
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec

class W2VWrapper(TextEmbedder):
    """
    Define a Poincare Embedding model.
    """
    def __init__(self, text_processor, w2v_params):
        self.text_processor = text_processor
        self.internal_model = Word2Vec(**w2v_params)

    @property
    def vocab_count(self):
        return self.internal_model.wv.vectors.shape[0]

    @property
    def emb_dim(self):
        return self.internal_model.wv.vectors.shape[1]

    @property
    def pad_index(self):
        return self.text_processor.pad_index

    def tokenize(self, text):
        return self.text_processor.tokenize(text)

    def index_tokens(self, tokens):
        return [
            self.internal_model.wv.vocab[token].index for token in tokens
            if token in self.internal_model.wv.vocab]

    def build_vocab(self, texts):
        self.internal_model.build_vocab(texts)

    def train(self, texts, **params):
        """
        Train this model.
        """
        self.internal_model.train(texts, **params)

    def embed(self, texts):
        padded_texts, _, _ = pad_texts(texts)
        return self.internal_model.wv.vectors[padded_texts]

    def get_word_emb(self):
        """
        Get Word Embeddings based on the text processor's dictionary this model has.
        """
        import numpy as np

        words = self.internal_model.wv.index2word

        word_index = self.text_processor.index_tokens(words)
        mask = np.zeros((self.text_processor.vocab_count,))
        mask[word_index] = 1

        word_emb = np.zeros((self.text_processor.vocab_count, self.emb_dim))
        word_emb[mask == 1] = self.internal_model.wv.vectors
        word_emb[mask == 0] = np.random.rand(np.sum(np.logical_not(mask)), self.emb_dim)

        return word_emb

    def before_serialize(self):
        self.text_processor.before_serialize()

    def after_deserialize(self):
        self.text_processor.after_deserialize()
