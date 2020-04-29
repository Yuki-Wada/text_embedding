"""
Define an Encoder-Decoder model.
"""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, backend, layers

logger = logging.getLogger(__name__)

class NaiveSeq2Seq(Model):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            emb_dim,
            enc_hidden_dim,
            dec_hidden_dim,
        ):
        super(NaiveSeq2Seq, self).__init__()

        self.encoder_embedding_layer = layers.Embedding(
            encoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.encoder_rnn = layers.LSTM(
            units=enc_hidden_dim,
            return_sequences=False,
            return_state=True,
        )

        self.decoder_embedding_layer = layers.Embedding(
            decoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.decoder_rnn = layers.LSTM(
            units=dec_hidden_dim,
            return_sequences=True,
            return_state=True,
        )
        self.output_layer = layers.Dense(decoder_vocab_count)

        encoder_inputs = layers.Input(shape=(None,))
        decoder_inputs = layers.Input(shape=(None,))
        self(encoder_inputs, decoder_inputs)

    def encode(self, encoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = self.encoder_embedding_layer.compute_mask(encoder_inputs)
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder_rnn(
            encoder_embedded, mask=encoder_masks)
        state = [encoder_state_h, encoder_state_c]

        return encoder_outputs, state, encoder_masks

    def decode(self, decoder_inputs, states):
        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_masks = self.decoder_embedding_layer.compute_mask(decoder_inputs)
        decoder_sequences, decoder_state_h, decoder_state_c = self.decoder_rnn(
            decoder_embedded,
            mask=decoder_masks,
            initial_state=states,
        )
        state = [decoder_state_h, decoder_state_c]

        decoder_outputs = self.output_layer(decoder_sequences)

        return decoder_outputs, state

    def call(self, encoder_inputs, decoder_inputs):
        _, state, _ = self.encode(encoder_inputs)
        decoder_outputs, _ = self.decode(decoder_inputs, state)

        return decoder_outputs

    def inference(self, encoder_inputs, begin_of_encode_index, seq_len):
        _, state, _ = self.encode(encoder_inputs)

        mb_size = encoder_inputs.shape[0]
        decoder_indices = np.empty((mb_size, seq_len + 1), dtype=np.int32)
        decoder_indices[:, 0] = begin_of_encode_index
        for i in range(seq_len):
            decoder_inputs = decoder_indices[:, i:i + 1]
            decoder_outputs, state = self.decode(decoder_inputs, state)
            decoder_probs = tf.nn.softmax(decoder_outputs, axis=2).numpy()[:, 0]
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            decoder_indices[:, i + 1] = np.array([
                np.random.choice(decoder_probs.shape[1], p=decoder_probs[j])
                for j in range(mb_size)
            ])

        return decoder_indices[:, 1:]

class Seq2SeqWithGlobalAttention(Model):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            emb_dim,
            enc_hidden_dim,
            dec_hidden_dim,
        ):
        super(Seq2SeqWithGlobalAttention, self).__init__()

        self.encoder_embedding_layer = layers.Embedding(
            encoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.encoder_rnn = layers.LSTM(
            units=enc_hidden_dim,
            return_sequences=True,
            return_state=True,
        )

        self.decoder_embedding_layer = layers.Embedding(
            decoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.decoder_rnn = layers.LSTM(
            units=dec_hidden_dim,
            return_sequences=True,
            return_state=True,
        )
        self.global_attention_layer = layers.Dense(enc_hidden_dim)
        self.concat_layer = layers.Concatenate()

        self.output_layer = layers.Dense(decoder_vocab_count)

        encoder_inputs = layers.Input(shape=(None,))
        decoder_inputs = layers.Input(shape=(None,))
        self(encoder_inputs, decoder_inputs)

    def encode(self, encoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = self.encoder_embedding_layer.compute_mask(encoder_inputs)
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder_rnn(
            encoder_embedded, mask=encoder_masks)
        state = [encoder_state_h, encoder_state_c]

        return encoder_outputs, state, encoder_masks

    def decode(self, decoder_inputs, states, encoder_outputs, encoder_masks):
        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_masks = self.decoder_embedding_layer.compute_mask(decoder_inputs)
        decoder_sequences, decoder_state_h, decoder_state_c = self.decoder_rnn(
            decoder_embedded,
            mask=decoder_masks,
            initial_state=states,
        )
        state = [decoder_state_h, decoder_state_c]

        scores = backend.batch_dot(
            self.global_attention_layer(decoder_sequences),
            backend.permute_dimensions(encoder_outputs, (0, 2, 1)),
        )
        scores -= backend.expand_dims(
            1 - backend.cast(encoder_masks, dtype='float32'), axis=1) * 1e18
        attentions = backend.softmax(scores, axis=2)
        weighted = backend.batch_dot(attentions, encoder_outputs)
        concat = self.concat_layer([decoder_sequences, weighted])

        decoder_outputs = self.output_layer(concat)

        return decoder_outputs, state

    def call(self, encoder_inputs, decoder_inputs):
        encoder_outputs, state, encoder_masks = self.encode(encoder_inputs)
        decoder_outputs, _ = self.decode(
            decoder_inputs, state, encoder_outputs, encoder_masks)

        return decoder_outputs

    def inference(self, encoder_inputs, begin_of_encode_index, seq_len):
        encoder_outputs, state, encoder_masks = self.encode(encoder_inputs)

        mb_size = encoder_inputs.shape[0]
        decoder_indices = np.empty((mb_size, seq_len + 1), dtype=np.int32)
        decoder_indices[:, 0] = begin_of_encode_index
        for i in range(seq_len):
            decoder_inputs = decoder_indices[:, i:i + 1]
            decoder_outputs, state = self.decode(
                decoder_inputs, state, encoder_outputs, encoder_masks)
            decoder_probs = tf.nn.softmax(decoder_outputs, axis=2).numpy()[:, 0]
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            decoder_indices[:, i + 1] = np.array([
                np.random.choice(decoder_probs.shape[1], p=decoder_probs[j])
                for j in range(mb_size)
            ])

        return decoder_indices

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

class TransformerEncoderBlock(Model):
    def __init__(
            self,
            model_dim,
            hidden_dim,
            head_count,
            feed_forward_hidden_dim,
        ):
        super(TransformerEncoderBlock, self).__init__()
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

class TransformerDecoderBlock(Model):
    def __init__(
            self,
            model_dim,
            hidden_dim,
            head_count,
            feed_forward_hidden_dim,
        ):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(
            model_dim, hidden_dim, head_count, forward_masked=True
        )
        self.layer_norm_after_masked_attention = layers.LayerNormalization()

        self.key_value_attention = MultiHeadAttention(
            model_dim, hidden_dim, head_count
        )
        self.layer_norm_after_key_value_attention = layers.LayerNormalization()

        self.hidden_layer = layers.Dense(feed_forward_hidden_dim)
        self.output_layer = layers.Dense(model_dim)
        self.layer_norm_after_feed_forward = layers.LayerNormalization()

    def call(self, inputs, key_value, input_mask=None, key_value_mask=None):
        h = self.masked_attention(inputs, inputs, inputs, input_mask)
        h = h + inputs
        h0 = self.layer_norm_after_masked_attention(h)

        h = self.key_value_attention(h0, key_value, key_value, key_value_mask)
        h = h + h0
        h0 = self.layer_norm_after_key_value_attention(h)

        h = self.hidden_layer(h0)
        h = self.output_layer(h)
        h = h + h0
        h = self.layer_norm_after_feed_forward(h)

        return h

class TransformerEncoderDecoder(Model):
    def __init__(
            self,
            encoder_vocab_count,
            decoder_vocab_count,
            emb_dim,
            encoder_hidden_dim,
            decoder_hidden_dim,
            head_count,
            feed_forward_hidden_dim,
            block_count,
        ):
        super(TransformerEncoderDecoder, self).__init__()

        self.positional_encoder = PositionalEncoder(emb_dim)

        self.encoder_embedding_layer = layers.Embedding(
            encoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.encoder_blocks = []
        for _ in range(block_count):
            self.encoder_blocks.append(TransformerEncoderBlock(
                emb_dim, encoder_hidden_dim, head_count, feed_forward_hidden_dim
            ))

        self.decoder_embedding_layer = layers.Embedding(
            decoder_vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.decoder_blocks = []
        for _ in range(block_count):
            self.decoder_blocks.append(TransformerDecoderBlock(
                emb_dim, decoder_hidden_dim, head_count, feed_forward_hidden_dim
            ))

        self.output_layer = layers.Dense(decoder_vocab_count)

        encoder_inputs = layers.Input(shape=(None,))
        decoder_inputs = layers.Input(shape=(None,))
        self(encoder_inputs, decoder_inputs)

    def encode(self, encoder_inputs):
        h = self.encoder_embedding_layer(encoder_inputs)
        if encoder_inputs.shape[1] is not None:
            h += tf.convert_to_tensor(self.positional_encoder(encoder_inputs.shape[1]))
        encoder_masks = self.encoder_embedding_layer.compute_mask(encoder_inputs)

        encoder_outputs = []
        for block in self.encoder_blocks:
            h = block(h, encoder_masks)
            encoder_outputs.append(h)

        return encoder_outputs, encoder_masks

    def decode(self, decoder_inputs, encoder_outputs, encoder_masks):
        h = self.decoder_embedding_layer(decoder_inputs)
        if decoder_inputs.shape[1] is not None:
            h += tf.convert_to_tensor(self.positional_encoder(decoder_inputs.shape[1]))
        decoder_masks = self.decoder_embedding_layer.compute_mask(decoder_inputs)

        for block, encoder_output in zip(
                self.decoder_blocks,
                encoder_outputs,
            ):
            h = block(h, encoder_output, decoder_masks, encoder_masks)

        h = self.output_layer(h)

        return h

    def call(self, encoder_inputs, decoder_inputs):
        encoder_outputs, encoder_masks = self.encode(encoder_inputs)
        decoder_outputs = self.decode(decoder_inputs, encoder_outputs, encoder_masks)

        return decoder_outputs

    def inference(self, encoder_inputs, begin_of_encode_index, seq_len):
        encoder_outputs, encoder_masks = self.encode(encoder_inputs)

        mb_size = encoder_inputs.shape[0]
        decoder_indices = np.empty((mb_size, seq_len + 1), dtype=np.int32)
        decoder_indices[:, 0] = begin_of_encode_index
        for i in range(seq_len):
            decoder_inputs = decoder_indices[:, :i + 1]
            decoder_outputs = self.decode(
                decoder_inputs, encoder_outputs, encoder_masks)
            decoder_probs = tf.nn.softmax(decoder_outputs[:, i:i + 1], axis=2).numpy()[:, 0]
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            decoder_indices[:, i + 1] = np.array([
                np.random.choice(decoder_probs.shape[1], p=decoder_probs[j])
                for j in range(mb_size)
            ])

        return decoder_indices

def decoder_loss(true, pred):
    is_effective = tf.cast(true != 0, tf.float32)
    weights = is_effective / (tf.math.reduce_sum(is_effective, axis=1, keepdims=True) + 1e-18)

    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses * weights, axis=1), axis=0)

    return entropy_loss
