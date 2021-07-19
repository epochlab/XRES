#!/usr/bin/env python3

import tensorflow as tf
from model.common import build_vgg

class lossModule:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.vgg = build_vgg((self.input_shape))
        self.mean_squared_error = tf.keras.losses.MeanSquaredError()
        self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

    @tf.function
    def content_loss(self, hr, sr):
        sr = tf.keras.applications.vgg19.preprocess_input(sr)
        hr = tf.keras.applications.vgg19.preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)
