"""
Define ResNet model.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Model, layers

class ConvUnit(Model):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(ConvUnit, self).__init__()

        self.conv = layers.Conv2D(filters, kernel_size, **kwargs)
        self.bn = layers.BatchNormalization(axis=-1)
        self.activation = layers.Activation(activation)

    def call(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.activation(h)

        return h

class BottleneckUnit(Model):
    def __init__(self, filters, kernel_size, **kwargs):
        super(BottleneckUnit, self).__init__()

        self.down_sample = ConvUnit(filters, 1)
        self.conv_1 = ConvUnit(
            filters, kernel_size, padding='same', **kwargs)
        self.conv_2 = ConvUnit(
            filters, kernel_size,
            padding='same', activation='linear', **kwargs)

    def call(self, x):
        h = self.conv_1(x)
        h = self.conv_2(h)

        h += self.down_sample(x)

        return h

class BottleneckBlock(Model):
    def __init__(self, filters, kernel_sizes):
        super(BottleneckBlock, self).__init__()

        self.units = []
        for kernel_size in kernel_sizes:
            self.units.append(BottleneckUnit(filters, kernel_size))
        self.pool = layers.MaxPooling2D(pool_size=2, strides=None, padding='valid')

    def call(self, x):
        h = x
        for unit in self.units:
            h = unit(h)
        h = self.pool(h)

        return h

class ResNet(Model):
    def __init__(self, label_count: int, input_shape: Tuple[int, int, int]):
        super(ResNet, self).__init__()

        params = [
            (64, (3, 3, 3)),
            (128, (3, 3, 3, 3)),
            (256, (3, 3, 3, 3, 3, 3)),
            (512, (3, 3, 3)),
        ]

        self.blocks = []
        for filters, kernel_sizes in params:
            self.blocks.append(BottleneckBlock(filters, kernel_sizes))

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(label_count)

        self(layers.Input(shape=input_shape))

    def call(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.flatten(h)
        h = self.fc(h)

        return h

def calc_loss(true, pred):
    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(entropy_losses)

    return entropy_loss
