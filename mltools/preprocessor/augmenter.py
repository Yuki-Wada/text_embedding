"""
Define an Image Augmenter Class.
"""
import numpy as np

class ImageAugmenter:
    def __init__(self):
        self.prob_to_flip_horizontal = 0.5
        self.prob_to_flip_vertical = 0.5

        self.prob_to_add_salt_pepper_noise = 0.3
        self.max_salt_pepper_noise = 0.02

        self.prob_to_adjust_gamma = 0.3
        self.min_gamma = 0.7

        self.prob_to_adjust_luminance = 0.3
        self.luminance_changes = np.array([-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]) / 255.0

    def __call__(self, image):
        if np.random.rand() < self.prob_to_flip_horizontal:
            image = self.flip_horizontal(image)
        if np.random.rand() < self.prob_to_flip_vertical:
            image = self.flip_vertical(image)
        if np.random.rand() < self.prob_to_add_salt_pepper_noise:
            image = self.add_salt_pepper_noise(image)
        if np.random.rand() < self.prob_to_adjust_gamma:
            image = self.adjust_gamma(image)
        if np.random.rand() < self.prob_to_adjust_luminance:
            image = self.adjust_luminance(image)

        return image

    def flip_horizontal(self, image):
        return image[:, ::-1]

    def flip_vertical(self, image):
        return image[::-1]

    def add_salt_pepper_noise(self, image):
        image += (np.random.rand(*image.shape) - 0.5) * 2 * self.max_salt_pepper_noise
        image = np.clip(image, 0.0, 1.0)
        return image

    def adjust_gamma(self, image):
        gamma = self.min_gamma + np.random.rand() * (1 - self.min_gamma)
        image = np.power(image, 1.0 / gamma)
        image = np.clip(image, 0.0, 1.0)

        return image

    def adjust_luminance(self, image):
        luminance_change = np.random.choice(self.luminance_changes)

        y = (0.2990 * image[2]) + (0.5870 * image[1]) + (0.1140 * image[0])
        cb = (-0.1687 * image[2]) + (-0.3312 * image[1]) + (0.50000 * image[0])
        cr = (0.50000 * image[2]) + (-0.4187 * image[1]) + (-0.0813 * image[0])

        y += luminance_change

        image[0] = y + (1.77200 * cb)
        image[1] = y + (-0.34414 * cb) + (-0.71414 * cr)
        image[2] = y + (1.40200 * cr)

        image = np.clip(image, 0.0, 1.0)

        return image
