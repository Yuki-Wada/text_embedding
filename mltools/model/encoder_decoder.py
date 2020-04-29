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

        decoder_outputs = self.output_layer(decoder_sequences)

        return decoder_outputs, [decoder_state_h, decoder_state_c]

    def call(self, encoder_inputs, decoder_inputs):
        _, state, _ = self.encode(encoder_inputs)
        decoder_outputs, _ = self.decode(decoder_inputs, state)

        return decoder_outputs

    def inference(self, encoder_inputs, begin_of_encode_index, seq_len):
        _, state, _ = self.encode(encoder_inputs)

        mb_size = encoder_inputs.shape[0]
        decoder_indices = np.empty((mb_size, seq_len), dtype=np.int32)
        decoder_inputs = np.ones((mb_size,), dtype=np.int32) * begin_of_encode_index
        for i in range(seq_len):
            decoder_inputs = np.expand_dims(decoder_inputs, axis=1)
            decoder_outputs, state = self.decode(decoder_inputs, state)
            decoder_probs = tf.nn.softmax(decoder_outputs, axis=2).numpy()[:, 0]
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            decoder_inputs = np.array([
                np.random.choice(decoder_probs.shape[1], p=decoder_probs[j])
                for j in range(mb_size)
            ])
            decoder_indices[:, i] = decoder_inputs

        return decoder_indices

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

        return decoder_outputs, [decoder_state_h, decoder_state_c]

    def call(self, encoder_inputs, decoder_inputs):
        encoder_outputs, state, encoder_masks = self.encode(encoder_inputs)
        decoder_outputs, _ = self.decode(
            decoder_inputs, state, encoder_outputs, encoder_masks)

        return decoder_outputs

    def inference(self, encoder_inputs, begin_of_encode_index, seq_len):
        encoder_outputs, state, encoder_masks = self.encode(encoder_inputs)

        mb_size = encoder_inputs.shape[0]
        decoder_indices = np.empty((mb_size, seq_len), dtype=np.int32)
        decoder_inputs = np.ones((mb_size,), dtype=np.int32) * begin_of_encode_index
        for i in range(seq_len):
            decoder_inputs = np.expand_dims(decoder_inputs, axis=1)
            decoder_outputs, state = self.decode(
                decoder_inputs, state, encoder_outputs, encoder_masks)
            decoder_probs = tf.nn.softmax(decoder_outputs, axis=2).numpy()[:, 0]
            decoder_probs = decoder_probs / np.sum(decoder_probs, axis=1, keepdims=True)
            decoder_inputs = np.array([
                np.random.choice(decoder_probs.shape[1], p=decoder_probs[j])
                for j in range(mb_size)
            ])
            decoder_indices[:, i] = decoder_inputs

        return decoder_indices

class TransformerEncoderDecoder(Model):
    def __init__(self):
        super(TransformerEncoderDecoder, self).__init__()

def decoder_loss(true, pred):
    is_effective = tf.cast(true != 0, tf.float32)
    weights = is_effective / (tf.math.reduce_sum(is_effective, axis=1, keepdims=True) + 1e-18)

    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses * weights, axis=1), axis=0)

    return entropy_loss
