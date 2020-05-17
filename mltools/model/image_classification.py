"""
Define image classification PyTorch models.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_activation(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'linear':
        return lambda x: x

def calc_loss(true, prob):
    return nn.NLLLoss()(nn.LogSoftmax(dim=1)(prob), true)

class ConvUnit(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            activation='relu', padding=0, **kwargs,
        ):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.activation(h)

        return h

class ResUnit(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            activation='relu', padding=0, **kwargs):
        super(ResUnit, self).__init__()

        if in_channels != out_channels:
            self.down_sample = ConvUnit(in_channels, out_channels, 1)
        else:
            self.down_sample = None

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            padding=padding, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = get_activation(activation)

    def forward(self, x):
        h = x

        if self.down_sample is not None:
            h0 = self.down_sample(x)
        else:
            h0 = x

        h = self.conv1(h)
        h = self.bn1(h)
        h = self.activation1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = h + h0
        h = self.activation2(h)

        return h

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, **kwargs):
        super(ResBlock, self).__init__()

        self.units = nn.ModuleList([])
        for i, kernel_size in enumerate(kernel_sizes):
            if i == 0:
                self.units.append(ResUnit(in_channels, out_channels, kernel_size, **kwargs))
            else:
                self.units.append(ResUnit(out_channels, out_channels, kernel_size, **kwargs))

    def forward(self, x):
        h = x
        for layer in self.units:
            h = layer(h)

        return h

class ResNet(nn.Module):
    def __init__(self, label_count: int, gpu_id=-1):
        super(ResNet, self).__init__()

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

        params = [
            (64, 64, 3, 2),
            (64, 128, 3, 2),
            (128, 256, 3, 2),
            (256, 512, 3, 2),
        ]

        self.conv1 = ConvUnit(3, 64, 7, stride=2)
        self.pool1 = nn.MaxPool2d(3, 2, padding=0)

        self.blocks = nn.ModuleList([])
        for in_channels, out_channels, kernel_size, block_count in params:
            self.blocks.append(
                ResBlock(
                    in_channels, out_channels, [kernel_size for _ in range(block_count)], padding=1,
                )
            )

        self.fc = nn.Linear(512, label_count)

        self.to(self.device)

    def forward(self, x):
        h = x

        h = self.conv1(h)
        h = self.pool1(h)

        for block in self.blocks:
            h = block(h)

        h = torch.mean(h, (2, 3))
        h = self.fc(h)

        return h
