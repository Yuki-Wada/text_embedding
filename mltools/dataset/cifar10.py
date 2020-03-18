from typing import Tuple
import logging
import numpy as np

import torch

logger = logging.getLogger(__name__)

class Cifar10DataSet:
    def __init__(self, image_npy_path: str, label_npy_path: str):
        self.images = np.load(image_npy_path)
        self.labels = np.load(label_npy_path)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError('The number of images should be equal to that of labels.')

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def image_shape(self) -> Tuple[int]:
        return self.images.shape[1:]

class Cifar10DataLoader:
    def __init__(self, data_set: Cifar10DataSet, mb_size: int):
        self.data_set = data_set
        self.mb_size = mb_size

    def __len__(self) -> int:
        return len(self.data_set)

    def get_iter(self):
        indices = np.arange(len(self.data_set))
        indices = np.random.permutation(indices)

        for i in range(0, len(self.data_set), self.mb_size):
            curr_mb_size = np.min([self.mb_size, len(self.data_set) - i])

            mb_images = np.empty([curr_mb_size] + list(self.data_set.image_shape))
            mb_labels = np.empty((curr_mb_size,))
            mb_images[:] = self.data_set.images[indices[i : i + self.mb_size]]
            mb_images /= 255.0
            mb_labels[:] = self.data_set.labels[indices[i : i + self.mb_size]]

            mb_images = mb_images.transpose(0, 3, 1, 2)
            mb_images = torch.Tensor(mb_images)
            mb_labels = torch.LongTensor(mb_labels)

            yield mb_images, mb_labels
