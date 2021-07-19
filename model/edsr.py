#!/usr/bin/env python3

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization, UpSampling2D, Activation, LeakyReLU, PReLU, Add, Dense, Flatten

def edsr_residual(x, num_filters, scaling):
    res = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    res = Conv2D(num_filters, 3, padding='same')(res)

    if scaling:
        res = Lambda(lambda t: t * scaling)(res)

    res = Add()([res, x])
    return res

def gridless_upsampling(model, num_filters, scale):
    def upsample(x, factor):
        x = UpSampling2D(size=factor)(model)
        x = Conv2D(num_filters, 3, padding='same')(x)
        return x

    if scale==2:
        model = upsample(model, 2)
    elif scale==3:
        model = upsample(model, 3)
    elif scale==4:
        model = upsample(model, 2)
        model = upsample(model, 2)

    return model

def build_edsr(input_shape, scale, num_filters, residual_blocks, res_block_scaling=0.1):
    input_layer = Input(shape=input_shape)

    gen1 = Conv2D(num_filters, kernel_size=9, padding='same')(input_layer)
    res = edsr_residual(gen1, num_filters, res_block_scaling)

    for i in range(residual_blocks - 1):
        res = edsr_residual(res, num_filters, res_block_scaling)

    gen2 = Conv2D(num_filters, kernel_size=3, padding='same')(res)
    model = Add()([gen2, gen1])

    model = gridless_upsampling(model, 256, scale)

    output = Conv2D(3, 9, padding='same')(model)
    output = Activation('tanh')(output)

    model = Model(inputs=[input_layer], outputs=[output], name='edsr_generator')
    return model

def discriminator_block(model, num_filters, kernel_size, strides):
    model = Conv2D(num_filters, kernel_size, strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    return model

def build_discriminator(input_shape, num_filters = 64):
    input_layer = Input(shape = (255,255,3))

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
