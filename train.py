#!/usr/bin/env python3

import os, math, random

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model.srgan import build_discriminator, build_srgan
from model.edsr import build_edsr
from data import dataIO
from loss import lossModule
from utils import generate_images, log_callback

print("Eager mode:", tf.executing_eagerly())

# -----------------------------

OUTDIR = 'metrics'

ROOT_0 = '/mnt/vanguard/datasets/vimeo_90k/toflow'
ROOT_1 = '/mnt/vanguard/datasets/ffhq-dataset/ffhq-512'
ROOT_2 = '/mnt/vanguard/datasets/celeba_bundle/data_hq_1024'

DIR_LIST = [ROOT_1, ROOT_2]

dataset = []
for ROOT in DIR_LIST:
    for path, subdirs, files in os.walk(ROOT):
        for name in sorted(files):
            filepath = os.path.join(path, name)
            dataset.append(filepath)
random.shuffle(dataset)

# -----------------------------

NETWORK = "EDSR"
RGB_MEAN = False
COCO = True

DELTA = 4
IMAGE_SHAPE = (256, 256, 3)
DOWNSAMPLE_SHAPE = (IMAGE_SHAPE[0]//DELTA, IMAGE_SHAPE[1]//DELTA, IMAGE_SHAPE[2])

BATCH_SIZE = 16
SPLIT_RATIO = 0.9
VALIDATION_SIZE = 100

RES_BLOCKS = 32
NUM_FILTERS = 256

EPOCHS = 300000

print("High Resolution Shape =", IMAGE_SHAPE)
print("Low Resolution Shape =", DOWNSAMPLE_SHAPE)

# -----------------------------

if RGB_MEAN:
    mean_array = dataIO.rgb_mean(IMAGE_SHAPE, dataset)

total_imgs = len(dataset)
split_index = int(math.floor(total_imgs) * SPLIT_RATIO)

n_train_imgs = dataset[:split_index]
n_test_imgs = dataset[split_index:-VALIDATION_SIZE]
n_val_imgs = dataset[total_imgs-VALIDATION_SIZE:]

# -----------------------------

dataIO = dataIO(DELTA, IMAGE_SHAPE)
loss = lossModule(IMAGE_SHAPE)

if NETWORK == "SRGAN":
    generator = build_srgan(DOWNSAMPLE_SHAPE)
if NETWORK == "EDSR":
    generator = build_edsr(DOWNSAMPLE_SHAPE, DELTA, NUM_FILTERS, RES_BLOCKS)

discriminator = build_discriminator(IMAGE_SHAPE)

generator_optimizer = Adam(0.0002, 0.5)
discriminator_optimizer = Adam(0.0002, 0.5)

@tf.function
def train_step(ds_low, ds_high):
    lr = tf.expand_dims(ds_low, axis=0)
    hr = tf.expand_dims(ds_high, axis=0)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr = generator(lr, training=True)
        hr_output = discriminator(hr, training=True)
        sr_output = discriminator(sr, training=True)

        con_loss = loss.content_loss(hr, sr)
        gen_loss = loss.generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = loss.discriminator_loss(hr_output, sr_output)

    generator_gradients = gen_tape.gradient(perc_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return con_loss, gen_loss, perc_loss, disc_loss

# -----------------------------

timestamp, summary_writer, checkpoint_prefix = log_callback(OUTDIR, generator, discriminator, generator_optimizer, discriminator_optimizer)
loss_min = 9999999

for epoch in tqdm(range(EPOCHS)):

    test_ds_low, test_ds_high = dataIO.sample_data(n_test_imgs, BATCH_SIZE, coco=False, rgb_mean=False)
    train_ds_low, train_ds_high = dataIO.sample_data(n_train_imgs, BATCH_SIZE, coco=True, rgb_mean=RGB_MEAN)

    generate_images(generator, test_ds_low, test_ds_high)

    for i in range(BATCH_SIZE):
        con_loss, gen_loss, perc_loss, disc_loss = train_step(train_ds_low[i], train_ds_high[i])

        with summary_writer.as_default():
            tf.summary.scalar('con_loss', con_loss, step=epoch)
            tf.summary.scalar('gen_loss', gen_loss, step=epoch)
            tf.summary.scalar('perc_loss', perc_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    if perc_loss < loss_min:
        generator.save(OUTDIR + "/results/generator_" + timestamp + '.h5')
        # print(" Model saved")
        loss_min = perc_loss

    if (epoch + 1) % 10000 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
