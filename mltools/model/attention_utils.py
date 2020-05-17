"""
Define utility classes for attention.
"""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, backend, layers

logger = logging.getLogger(__name__)

class PositionalEncoder:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim

    def __call__(self, length):
        dim_indices = np.arange(self.emb_dim)
        x = np.outer(
            np.arange(length),
            10000 ** (-2 * np.floor(dim_indices / 2) / self.emb_dim),
        )
        positional_encoding = np.sin(x + np.pi / 2 * dim_indices % 2)
        positional_encoding = positional_encoding.astype(np.float32)

        return positional_encoding

class SingleDotProductAttention:
    def __init__(self, forward_masked=False):
        self.forward_masked = forward_masked

    def __call__(self, query, key, value, key_mask=None):
        scores = backend.batch_dot(query, backend.permute_dimensions(key, (0, 2, 1)))
        scores /= key.shape[2] ** 0.5
        if key_mask is not None:
            scores -= backend.expand_dims(
                1 - backend.cast(key_mask, dtype='float32'), axis=1) * 1e18
        if self.forward_masked and query.shape[1] is not None:
            scores -= (1 - tf.linalg.band_part(
                tf.ones((query.shape[1], query.shape[1])), -1, 0
            )) * 1e18
        attentions = backend.softmax(scores, axis=2)
        outputs = backend.batch_dot(attentions, value)

        return outputs

class MultiHeadAttention(Model):
    def __init__(self, model_dim, hidden_dim, head_count, forward_masked=False):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % head_count == 0, \
            'The argument model_dim should be divided by the argument head_count.'

        self.head_count = head_count

        self.query_dense_layers = []
        self.key_dense_layers = []
        self.value_dense_layers = []
        self.dot_product_attentions = []
        for _ in range(head_count):
            self.query_dense_layers.append(layers.Dense(hidden_dim))
            self.key_dense_layers.append(layers.Dense(hidden_dim))
            self.value_dense_layers.append(layers.Dense(hidden_dim))
            self.dot_product_attentions.append(
                SingleDotProductAttention(forward_masked=forward_masked))

        self.concat_layer = layers.Concatenate()
        self.output_layer = layers.Dense(model_dim)

    def call(self, query, key, value, key_mask=None):
        attention_outputs = []
        split_queries = tf.split(query, num_or_size_splits=self.head_count, axis=-1)
        split_keys = tf.split(key, num_or_size_splits=self.head_count, axis=-1)
        split_values = tf.split(value, num_or_size_splits=self.head_count, axis=-1)

        for query_dense, key_dense, value_dense, split_query, split_key, split_value, attention_layer in zip(
                self.query_dense_layers,
                self.key_dense_layers,
                self.value_dense_layers,
                split_queries,
                split_keys,
                split_values,
                self.dot_product_attentions,
            ):
            h = attention_layer(
                query_dense(split_query), key_dense(split_key), value_dense(split_value), key_mask)
            attention_outputs.append(h)

        concat = self.concat_layer(attention_outputs)
        outputs = self.output_layer(concat)

        return outputs

class Transformer(Model):
    def __init__(
            self,
            model_dim,
            hidden_dim,
            head_count,
            feed_forward_hidden_dim,
        ):
        super(Transformer, self).__init__()
        self.attention = MultiHeadAttention(
            model_dim, hidden_dim, head_count
        )
        self.layer_norm_after_attention = layers.LayerNormalization()

        self.hidden_layer = layers.Dense(feed_forward_hidden_dim)
        self.output_layer = layers.Dense(model_dim)
        self.layer_norm_after_feed_forward = layers.LayerNormalization()

    def call(self, x, mask=None):
        h = self.attention(x, x, x, mask)
        h = h + x
        h0 = self.layer_norm_after_attention(h)

        h = self.hidden_layer(h0)
        h = self.output_layer(h)
        h = h + h0
        h = self.layer_norm_after_feed_forward(h)

        return h
