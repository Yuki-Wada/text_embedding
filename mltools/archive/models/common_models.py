"""
Define common models in this module.
"""
import numpy as np

import torch
import torch.nn as nn

from nlp_model.constant import INF

class PositionalEncoding:
    """
    Define the class which creates a positional encoding of an input length.
    """
    def __init__(self, input_dim, use_cuda=False):
        if input_dim % 2 != 0:
            raise ValueError(
                'You must need the input dimension to be even'
                'when using a PositionalEncoding class.')
        self.input_dim = input_dim
        self.use_cuda = use_cuda

    def encode(self, seq_len):
        """
        Encode an positional encoding of an input length.
        """
        sin_length = int((self.input_dim + 1) / 2)
        cos_length = self.input_dim - sin_length
        pos_enc = np.zeros((seq_len, self.input_dim))

        pos_enc[:, :sin_length] = np.sin(np.matmul(
            np.arange(seq_len).reshape(-1, 1),
            10000 ** (- np.arange(sin_length).reshape(1, -1) * 2 / self.input_dim)))

        pos_enc[:, sin_length:] = np.cos(np.matmul(
            np.arange(seq_len).reshape(-1, 1),
            10000 ** (- np.arange(cos_length).reshape(1, -1) * 2 / self.input_dim)))

        pos_enc = torch.Tensor(pos_enc)
        if self.use_cuda:
            pos_enc = pos_enc.cuda()

        return pos_enc

    def before_serialize(self):
        """
        You must execute this function before serializing this model.
        """
        self.use_cuda = False

    def after_deserialize(self, use_cuda=False):
        """
        You must execute this function after deserializing into this model.
        """
        self.use_cuda = use_cuda

class SelfAttentionLayer(nn.Module):
    """
    Define a model which sums an input vector by the weights the model calculates itself.
    """
    def __init__(self, input_dim, hidden_dim, attention_dim, use_cuda=False):
        super(SelfAttentionLayer, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, attention_dim)
        self.activate2 = nn.Softmax(dim=1)
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.cuda()

    def forward(self, vectors, masks):
        batch_count = vectors.shape[0]

        curr = self.linear1(vectors)
        curr = self.activate1(curr)
        curr = self.linear2(curr)
        curr -= torch.unsqueeze(masks * INF, 2)
        attentions = self.activate2(curr)

        results = torch.matmul(vectors.transpose(1, 2), attentions).view(batch_count, -1)

        return results, attentions
