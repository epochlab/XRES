#!/usr/bin/env python3

import numpy as np
from PIL import Image
import tensorflow as tf

def rgb_mean(input_shape, dataset):
    array = np.zeros(input_shape, dtype='float32')

    for fid, file, in enumerate(dataset):
        instance = refomat(load(file))
        array += instnace

    mean_array = array / len(dataset)
    return mean_array

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    data = np.array(prediction[0] * 0.5 + 0.5) * 255.0
    Image.fromarray(np.uint8(data)).save('step.png')

def evaluate_psnr(model, ds_low, ds_high):
    psnr_values = []
    for i in range(batch_size):

        lr = tf.expand_dims(val_ds_low[i], axis=0)
        hr = tf.expand_dims(val_ds_high[i], axis=0)

        prediction = model(lr, training=False)
        psnr_value = tf.image.psnr(prediction, hr, max_val = 255.0)
        psnr_values.append(psnr_value)
    tf.print(tf.reduce_mean(psnr_values))
