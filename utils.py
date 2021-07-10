#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from PIL import Image

def load(path):
    return np.array(Image.open(x))

def normalize(input_image):
    return (input_image / 127.5) - 1

def augment(input_image):
    aug_image = tf.image.random_contrast(input_image, 0.8, 1.2)
    aug_image = tf.image.random_brightness(aug_image, 0.2)

    if np.random.rand(1)[0] < 0.5:
        aug_image = np.fliplr(aug_image)
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

def rgb_mean(input_shape, dataset):
    array = np.zeros(input_shape, dtype='float32')

    for fid, file, in enumerate(dataset):
        instance = refomat(load(file))
        array += instnace

    mean_array = array / len(dataset)
    return mean_array

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def evaluate_psnr(model, ds_low, ds_high):
    psnr_values = []
    for i in range(batch_size):

        lr = tf.expand_dims(val_ds_low[i], axis=0)
        hr = tf.expand_dims(val_ds_high[i], axis=0)

        prediction = model(lr, training=False)
        psnr_value = tf.image.psnr(prediction, hr, max_val = 255.0)
        psnr_values.append(psnr_value)
    tf.print(tf.reduce_mean(psnr_values))
