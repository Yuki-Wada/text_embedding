"""
Define a Data Set Class to Preprocess Cifar-10.
"""

from typing import Tuple
import logging
import numpy as np

from mltools.preprocessor.augmenter import ImageAugmenter

logger = logging.getLogger(__name__)

class Cifar10DataSet:
    def __init__(self, image_npy_path: str, label_npy_path: str):
        self.images = np.load(image_npy_path).astype(np.float32) / 255
        self.labels = np.load(label_npy_path).astype(np.int32)
        if len(self.images.shape) < 4:
            self.images = np.expand_dims(self.images, axis=-1)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError('The number of images should be equal to that of labels.')

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def image_shape(self) -> Tuple[int]:
        return self.images.shape[1:]

class Cifar10DataLoader:
    def __init__(self, data_set: Cifar10DataSet, mb_size: int, use_augment=False):
        self.data_set = data_set
        self.mb_size = mb_size
        if use_augment:
            self.augmenter = ImageAugmenter()
        else:
            self.augmenter = None

    @property
    def iter_count(self):
        return int((len(self) + self.mb_size - 1) / self.mb_size)

    def __len__(self) -> int:
        return len(self.data_set)

    def __iter__(self):
        indices = np.arange(len(self.data_set))
        indices = np.random.permutation(indices)

        for i in range(0, len(self.data_set), self.mb_size):
            mb_images = self.data_set.images[indices[i : i + self.mb_size]]
            mb_labels = self.data_set.labels[indices[i : i + self.mb_size]]

            if self.augmenter is not None:
                for mb_idx in range(mb_images.shape[0]):
                    mb_images[mb_idx] = self.augmenter(mb_images[mb_idx])

            mb_images = mb_images.transpose(0, 3, 1, 2)

            yield mb_images, mb_labels
