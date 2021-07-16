#!/usr/bin/env python3

from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19

def build_vgg(input_shape):
    vgg = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
    return Model(vgg.input, vgg.layers[9].output)
