"""
Define a transformer model which assign a label to an input text.
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from mltools.model.attention_utils import PositionalEncoder, MultiHeadAttention

class MaskedTransformerBlock(Model):
    def __init__(
            self,
            model_dim,
            hidden_dim,
            head_count,
            feed_forward_hidden_dim,
        ):
        super(MaskedTransformerBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(
            model_dim, hidden_dim, head_count, forward_masked=True
        )
        self.layer_norm_after_attention = layers.LayerNormalization()

        self.hidden_layer = layers.Dense(feed_forward_hidden_dim)
        self.output_layer = layers.Dense(model_dim)
        self.layer_norm_after_feed_forward = layers.LayerNormalization()

    def call(self, inputs, input_mask=None):
        h = self.masked_attention(inputs, inputs, inputs, input_mask)
        h = h + inputs
        h0 = self.layer_norm_after_attention(h)

        h = self.hidden_layer(h0)
        h = self.output_layer(h)
        h = h + h0
        h = self.layer_norm_after_feed_forward(h)

        return h

class GPT(Model):
    """
    Define a model to pretrain GPT.
    """
    def __init__(
            self,
            vocab_count,
            emb_dim,
            transformer_hidden_dim,
            head_count,
            feed_forward_hidden_dim,
            block_count,
        ):
        super(GPT, self).__init__()

        self.embedding_layer = layers.Embedding(
            vocab_count,
            emb_dim,
            mask_zero=True,
        )
        self.positional_encoder = PositionalEncoder(emb_dim)

        self.blocks = []
        for _ in range(block_count):
            self.blocks.append(MaskedTransformerBlock(
                emb_dim, transformer_hidden_dim, head_count, feed_forward_hidden_dim
            ))

        self.output_layer = layers.Dense(self.vocab_count)

        inputs = layers.Input(shape=(None,))
        self(inputs)

    def embed(self, inputs):
        h = self.embedding_layer(inputs)
        if inputs.shape[1] is not None:
            h += tf.convert_to_tensor(self.positional_encoder(inputs.shape[1]))
        masks = self.embedding_layer.compute_mask(inputs)

        for block in self.blocks:
            h = block(h, masks)

        return h

    def decode(self, embedded):
        return self.output_layer(embedded)

    def call(self, inputs):
        h = self.embed(inputs)
        h = self.decode(h)

        return h

class GPTFineTuner(Model):
    """
    Define a model to finetune GPT.
    """
    def __init__(
            self,
            pretrained_gpt_model,
            target_model,
        ):
        super(GPTFineTuner, self).__init__()

        self.gpt_model = pretrained_gpt_model
        self.target_model = target_model

        inputs = layers.Input(shape=(None,))
        self(inputs)

    def embed(self, inputs):
        return self.gpt_model.embed(inputs)

    def decode(self, embedded):
        return self.gpt_model.output_layer(embedded)

    def call(self, inputs):
        """
        Finetune this model.
        """
        h = self.gpt_model.embed(inputs)
        target = self.target_model(h)
        decoded = self.gpt_model.decode(h)

        return target, decoded

def pretrain_loss(true, pred, pad_index=0):
    is_effective = tf.cast(true != pad_index, tf.float32)
    weights = is_effective / (tf.math.reduce_sum(is_effective, axis=1, keepdims=True) + 1e-18)

    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses * weights, axis=1), axis=0)

    return entropy_loss
