#!/usr/bin/env python3

import tensorflow as tf

mean_squared_error = tf.keras.losses.MeanSquaredError()
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def generator_loss(sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out)

def discriminator_loss(hr_out, sr_out):
    hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)
    sr_loss = binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
    return hr_loss + sr_loss
