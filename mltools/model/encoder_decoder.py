"""
Define a transformer model which assign a label to an input text.
"""
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow as tf

class Encoder(nn.Module):
    """
    Define a pretraining GPT model.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding_layer = layers.Embedding(1000, 6)

    def forward(self, texts):
        pass


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
