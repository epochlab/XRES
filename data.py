#!/usr/bin/env python3

from PIL import Image
from tqdm import tqdm

import numpy as np
import tensorflow as tf

class dataIO:
    def __init__(self, scale, image_shape):
        self.high_resolution_shape = image_shape
        self.low_resolution_shape = (image_shape[0]//scale, image_shape[1]//scale, image_shape[2])

    def load(self, file):
        img = Image.open(file)
        return np.array(img, dtype='float32')

    def normalize(self, input_image):
        n_image = (input_image / 127.5) - 1
        return n_image

    def augment(self, input_image):
        aug_image = input_image

        aug_image = tf.image.random_contrast(aug_image, 0.8, 1.2)
        aug_image = tf.image.random_brightness(aug_image, 0.2)

        if np.random.random_sample() < 0.5:
            img = tf.image.flip_left_right(aug_image)

        return aug_image

    def reformat(self, input_image):
        if input_image.shape[0] > input_image.shape[1]:
            align = 'portraint'
            factor = input_image.shape[1] / self.high_resolution_shape[0]
            reformat_image = tf.image.resize(input_image, size = [int(input_image.shape[0] / factor), self.high_resolution_shape[0]])
        elif input_image.shape[0] < input_image.shape[1]:
            align = 'landscape'
            factor = input_image.shape[0] / self.high_resolution_shape[0]
            reformat_image = tf.image.resize(input_image, size = [self.high_resolution_shape[0], int(input_image.shape[1] / factor)])
        else:
            align = 'square'
            factor = input_image.shape[0] / self.high_resolution_shape[0]
            reformat_image = tf.image.resize(input_image, size = [self.high_resolution_shape[0], self.high_resolution_shape[1]])

        if align != 'square':
            reformat_image = tf.image.random_crop(reformat_image, size=[self.high_resolution_shape[0], self.high_resolution_shape[1], 3])

        return reformat_image

    def resize(self, input_image):
        low_resolution_image = tf.image.resize(input_image, self.low_resolution_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        high_resolution_image = tf.image.resize(input_image, self.high_resolution_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return low_resolution_image, high_resolution_image

    def sample_data(self, data, batch_size, mean_array, coco, rgb_mean):
        img_batch = np.random.choice(data, size=batch_size)

        ds_low = []
        ds_high = []

        for i, index in enumerate(img_batch):
            input_image = self.load(index)
            x_image = self.reformat(input_image)
            n_image = self.normalize(x_image)

            if rgb_mean:
                n_image -= self.normalize(mean_array)

            if coco:
                n_image = self.augment(n_image)

            low_resolution_image, high_resolution_image = self.resize(n_image)

            ds_low.append(low_resolution_image)
            ds_high.append(high_resolution_image)

        ds_low = tf.convert_to_tensor(ds_low)
        ds_high = tf.convert_to_tensor(ds_high)
        return ds_low, ds_high

    def rgb_mean(self, input_shape, dataset):
        array = np.zeros(input_shape, dtype='float32')

        for file in tqdm(dataset, desc="RGB Mean"):
            instance = self.reformat(self.load(file))
            array += instance

        mean_array = array / len(dataset)
        return mean_array
