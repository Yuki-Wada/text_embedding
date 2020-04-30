"""
Define image classification models.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Model, layers, backend

class ConvUnit(Model):
    def __init__(self, filters, kernel_size, activation='relu', padding='same', **kwargs):
        super(ConvUnit, self).__init__()

        self.conv = layers.Conv2D(filters, kernel_size, padding=padding, **kwargs)
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.activation(h)

        return h

class ShakeShakeUnit(Model):
    def __init__(self, filters, kernel_size, activation='relu', padding='same', **kwargs):
        super(ShakeShakeUnit, self).__init__()

        self.conv1 = ConvUnit(filters, kernel_size, activation='linear', padding=padding, **kwargs)
        self.conv2 = ConvUnit(filters, kernel_size, activation='linear', padding=padding, **kwargs)
        self.activation = layers.Activation(activation)

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # create alpha and beta
        batch_size = backend.shape(x1)[0]
        alpha = backend.random_uniform((batch_size, 1, 1, 1))
        beta = backend.random_uniform((batch_size, 1, 1, 1))

        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + backend.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)

        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return self.activation(backend.in_train_phase(x_shake, x_even))

class ConvBlock(Model):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super(ConvBlock, self).__init__()

        self.units = []
        for kernel_size in kernel_sizes:
            self.units.append(ConvUnit(filters, kernel_size, **kwargs))

        self.dropout = layers.Dropout(0.25)

    def call(self, x):
        h = x
        for layer in self.units:
            h = layer(h)

        h = self.dropout(h)

        return h

class ConvNet(Model):
    def __init__(self, label_count: int, input_shape: Tuple[int, int, int]):
        super(ConvNet, self).__init__()

        params = [
            (64, [1, 3, 5]),
            (128, [1, 3, 5]),
            (256, [1, 3, 5]),
        ]

        self.blocks = []
        for i, (filters, kernel_sizes) in enumerate(params):
            self.blocks.append(
                ConvBlock(filters, kernel_sizes, padding='valid', kernel_initializer='he_normal'))
            if i == 0:
                self.blocks.append(layers.MaxPool2D((2, 2)))

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

class ResUnit(Model):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResUnit, self).__init__()

        self.down_sample = ConvUnit(filters, 1)
        self.conv_1 = ConvUnit(filters, kernel_size, **kwargs)
        self.conv_2 = ConvUnit(filters, kernel_size, activation='linear', **kwargs)
        self.activation = layers.Activation('relu')

    def call(self, x):
        h = self.conv_1(x)
        h = self.conv_2(h)

        if x.shape[3] != h.shape[3]:
            h0 = self.down_sample(x)
        else:
            h0 = x
        h = h + h0

        h = self.activation(h)

        return h

class ResBlock(Model):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super(ResBlock, self).__init__()

        self.units = []
        for kernel_size in kernel_sizes:
            self.units.append(ResUnit(filters, kernel_size, **kwargs))

    def call(self, x):
        h = x
        for layer in self.units:
            h = layer(h)

        return h

class ResNet(Model):
    def __init__(self, label_count: int, input_shape: Tuple[int, int, int]):
        super(ResNet, self).__init__()

        params = [
            (64, 3, 2),
            (128, 3, 2),
            (256, 3, 2),
            (512, 3, 2),
        ]

        self.conv1 = ConvUnit(64, 7, strides=2, kernel_initializer='he_normal')

        self.blocks = []
        for filters, kernel_size, block_count in params:
            self.blocks.append(
                ResBlock(filters, [kernel_size for _ in range(block_count)], kernel_initializer='he_normal'))

        self.fc = layers.Dense(label_count, kernel_initializer='he_normal')

        self(layers.Input(shape=input_shape))

    def call(self, x):
        h = self.conv1(x)
        h = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(h)
        for block in self.blocks:
            h = block(h)

        h = layers.GlobalAveragePooling2D()(h)
        h = self.fc(h)

        return h

def calc_loss(true, pred):
    entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(true, pred, name=None)
    entropy_loss = tf.reduce_mean(entropy_losses)

    return entropy_loss
