"""
Define Encoder-Decoder models.
"""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, backend, layers

from mltools.model.attention_utils import PositionalEncoder, MultiHeadAttention, \
    Transformer as TransformerEncoderBlock

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

def decoder_loss(true, pred, pad_index=0):
    is_effective = tf.cast(true != pad_index, tf.float32)
    weights = is_effective / (tf.math.reduce_sum(is_effective, axis=1, keepdims=True) + 1e-18)

    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses * weights, axis=1), axis=0)

    return entropy_loss
