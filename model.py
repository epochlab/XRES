#!/usr/bin/env python3

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, LeakyReLU, PReLU, Add, Dense, Flatten, Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

r = 4
image_shape = (256, 256, 3)
downsample_shape = (image_shape[0]//r, image_shape[1]//r, image_shape[2])
low_resolution_shape = downsample_shape
high_resolution_shape = image_shape

def residual_block(x):
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = BatchNormalization(momentum=momentum)(res)
    res = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(res)
    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res

def upsampling_block(model, kernal_size, filters, strides):
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

    return model
def build_generator():

    residual_blocks = 16
    momentum = 0.8
    input_shape = low_resolution_shape

    input_layer = Input(shape=input_shape)

    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(input_layer)
    gen1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(gen1)

    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    model = Add()([gen2, gen1])

    for index in range(2):
        model = upsampling_block(model, 3, 256, 1)

    output = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(model)
    output = Activation('tanh')(output)

    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model

def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

def build_discriminator(num_filters = 64):
    input_shape = high_resolution_shape
    input_layer = Input(shape = input_shape)

    dis1 = Conv2D(num_filters, 3, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha = 0.2)(dis1)

    dis2 = discriminator_block(dis1, num_filters, 3, 2)

    dis3 = discriminator_block(dis2, num_filters * 2, 3, 1)
    dis4 = discriminator_block(dis3, num_filters * 2, 3, 2)

    dis5 = discriminator_block(dis4, num_filters * 4, 3, 1)
    dis6 = discriminator_block(dis5, num_filters * 4, 3, 2)

    dis7 = discriminator_block(dis6, num_filters * 8, 3, 1)
    dis8 = discriminator_block(dis7, num_filters * 8, 3, 2)

    dis9 = Flatten()(dis8)
    dis9 = Dense(1024)(dis9)
    dis9 = LeakyReLU(alpha = 0.2)(dis9)

    output = Dense(units=1)(dis9)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')

    return model

def build_vgg():
    vgg = VGG19(input_shape=(high_resolution_shape), weights='imagenet', include_top=False)
    return Model(vgg.input, vgg.layers[9].output)
