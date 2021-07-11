#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

r = 4

image_shape = (256, 256, 3)
downsample_shape = (image_shape[0]//r, image_shape[1]//r, image_shape[2])

low_resolution_shape = downsample_shape
high_resolution_shape = image_shape

def load(file):
    file = tf.io.read_file(file)
    image = tf.image.decode_png(file)
    data = tf.cast(image, tf.float32)

    return data

def normalize(input_image):
    n_image = (input_image / 127.5) - 1

    return n_image

def augment(input_image):
    aug_image = input_image

    aug_image = tf.image.random_contrast(aug_image, 0.8, 1.2)
    aug_img = tf.image.random_brightness(aug_image, 0.2)

    if tf.random.uniform(shape=[]) < 0.5:
        img = tf.image.flip_left_right(aug_image)

    return aug_image

def reformat(input_image):
    if input_image.shape[0] > input_image.shape[1]:
        align = 'portraint'
        factor = input_image.shape[1] / high_resolution_shape[0]
        reformat_image = tf.image.resize(input_image, size = [int(input_image.shape[0] / factor), high_resolution_shape[0]])
    elif input_image.shape[0] < input_image.shape[1]:
        align = 'landscape'
        factor = input_image.shape[0] / high_resolution_shape[0]
        reformat_image = tf.image.resize(input_image, size = [high_resolution_shape[0], int(input_image.shape[1] / factor)])
    else:
        align = 'square'
        factor = input_image.shape[0] / high_resolution_shape[0]
        reformat_image = tf.image.resize(input_image, size = [high_resolution_shape[0], high_resolution_shape[1]])

    if align != 'square':
        reformat_image = tf.image.random_crop(reformat_image, size=[high_resolution_shape[0], high_resolution_shape[1], 3])

    return reformat_image

def resize(input_image):
    low_resolution_image = tf.image.resize(input_image, low_resolution_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    high_resolution_image = tf.image.resize(input_image, high_resolution_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return low_resolution_image, high_resolution_image

def sample_data(data, batch_size, coco, rgb_mean):
    img_batch = np.random.choice(data, size=batch_size)

    ds_low = []
    ds_high = []

    for i, index in enumerate(img_batch):
        input_image = load(index)
        x_image = reformat(input_image)
        n_image = normalize(x_image)

        if coco:
            n_image = augment(n_image)

        low_resolution_image, high_resolution_image = resize(n_image)

        ds_low.append(low_resolution_image)
        ds_high.append(high_resolution_image)

    ds_low = tf.convert_to_tensor(ds_low)
    ds_high = tf.convert_to_tensor(ds_high)

    return ds_low, ds_high
