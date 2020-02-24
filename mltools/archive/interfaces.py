"""
Define interface classes in this module.
"""
from abc import ABC, abstractmethod
import dill
import numpy as np

class BaseObject(ABC):
    """
    Define a base object class in this module.
    """
    @abstractmethod
    def before_serialize(self):
        """
        Execute this function before serializing this object.
        """

    @abstractmethod
    def after_deserialize(self, **params):
        """
        Execute this function after deserializing this object.
        """

    def save(self, save_path):
        """
        Serialize this object.
        """
        params = self.before_serialize()
        params = {} if params is None else params
        with open(save_path, 'wb') as _:
            dill.dump(self, _)
        self.after_deserialize(**params)

    @classmethod
    def load(cls, load_path):
        """
        Deserialize an object from a file.
        """
        with open(load_path, 'rb') as _:
            model = dill.load(_)
        assert isinstance(model, cls), \
            'Input appropriate {} model file path.'.format(cls.__name__)
        model.after_deserialize()
        return model

class TextModel(BaseObject):
    """
    Define a interface model which processes text data.
    """
    @property
    @abstractmethod
    def vocab_count(self):
        """
        Return the count of vocabularies used.
        """

    @property
    @abstractmethod
    def emb_dim(self):
        """
        Return the embedding dimension used in this model.
        """

    @property
    @abstractmethod
    def pad_index(self):
        """
        Return a PAD token's index used in this model.
        """

    @abstractmethod
    def tokenize(self, text):
        """
        Tokenize a text.
        """

    @abstractmethod
    def index_tokens(self, tokens):
        """
        Index tokens.
        """

class TextEmbedder(TextModel):
    """
    Define a interface model which learns word embeddings.
    """
    @abstractmethod
    def embed(self, texts, **params):
        """
        Embed input texts.
        """

    @staticmethod
    def convert(embeds, weights=None):
        """
        Convert input texts into vectors (with a fixed dimension).
        """
        return np.sum(np.expand_dims(weights, 2) * embeds, axis=1)

class TextClassifier(TextModel):
    """
    Define a interface model which classify each text appropriately.
    """
    @abstractmethod
    def predict(self, texts):
        """
        Predict input texts.
        """
