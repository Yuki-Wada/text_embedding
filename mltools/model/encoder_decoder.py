"""
Define a transformer model which assign a label to an input text.
"""
import tensorflow as tf
from tensorflow.keras import Model, backend, layers

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

    def call(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        _, encoder_state_h, encoder_state_c = self.encoder_rnn(encoder_embedded)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_sequences, _, _ = self.decoder_rnn(
            decoder_embedded,
            initial_state=encoder_states,
        )
        decoder_outputs = self.output_layer(decoder_sequences)

        return decoder_outputs

    def get_decoder_outputs(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        _, encoder_state_h, encoder_state_c = self.encoder_rnn(encoder_embedded)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_sequences, _, _ = self.decoder_rnn(
            decoder_embedded,
            initial_state=encoder_states,
        )
        decoder_outputs = self.output_layer(decoder_sequences)
        return decoder_outputs

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

    def call(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = self.encoder_embedding_layer.compute_mask(encoder_inputs)
        encoder_masks = 1 - backend.cast(encoder_masks, dtype='float32')
        encoder_sequences, encoder_state_h, encoder_state_c = self.encoder_rnn(encoder_embedded)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_sequences, _, _ = self.decoder_rnn(
            decoder_embedded,
            initial_state=encoder_states,
        )

        scores = backend.batch_dot(
            self.global_attention_layer(decoder_sequences),
            backend.permute_dimensions(encoder_sequences, (0, 2, 1)),
        )
        scores -= backend.expand_dims(encoder_masks, axis=1) * 1e18
        attentions = backend.softmax(scores, axis=2)
        weighted = backend.batch_dot(attentions, encoder_sequences)
        concat = self.concat_layer([decoder_sequences, weighted])

        decoder_outputs = self.output_layer(concat)
        return decoder_outputs

    def get_decoder_outputs(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding_layer(encoder_inputs)
        encoder_masks = self.encoder_embedding_layer.compute_mask(encoder_inputs)
        encoder_masks = 1 - backend.cast(encoder_masks, dtype='float32')
        encoder_sequences, encoder_state_h, encoder_state_c = self.encoder_rnn(encoder_embedded)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_embedded = self.decoder_embedding_layer(decoder_inputs)
        decoder_sequences, _, _ = self.decoder_rnn(
            decoder_embedded,
            initial_state=encoder_states,
        )

        scores = backend.batch_dot(
            self.global_attention_layer(decoder_sequences),
            backend.permute_dimensions(encoder_sequences, (0, 2, 1)),
        )
        scores -= backend.expand_dims(encoder_masks, axis=1) * 1e18
        attentions = backend.softmax(scores, axis=2)
        weighted = backend.batch_dot(attentions, encoder_sequences)
        concat = self.concat_layer([decoder_sequences, weighted])

        decoder_outputs = self.output_layer(concat)
        return decoder_outputs

def decoder_loss(true, pred):
    is_effective = tf.cast(true != 0, tf.float32)
    weights = is_effective / (tf.math.reduce_sum(is_effective, axis=1, keepdims=True) + 1e-18)

    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses * weights, axis=1), axis=0)

    return entropy_loss
