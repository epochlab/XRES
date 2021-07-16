#!/usr/bin/env python3

import os, datetime

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    data = np.array(prediction[0] * 0.5 + 0.5) * 255.0
    Image.fromarray(np.uint8(data)).save('step.png')

def log_callback(outdir, generator, discriminator, generator_optimizer, discriminator_optimizer):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("ID:", timestamp)

    log_dir = outdir + "/logs"
    summary_writer = tf.summary.create_file_writer(log_dir + "/fit/" + timestamp)

    checkpoint_dir = outdir + '/training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    command = "tensorboard --logdir=" + str(log_dir) + " --port=6006 &"
    os.system(command)

    return timestamp, summary_writer, checkpoint_prefix

def evaluate_psnr(model, ds_low, ds_high):
    psnr_values = []
    for i in range(batch_size):

        lr = tf.expand_dims(val_ds_low[i], axis=0)
        hr = tf.expand_dims(val_ds_high[i], axis=0)

        prediction = model(lr, training=False)
        psnr_value = tf.image.psnr(prediction, hr, max_val = 255.0)
        psnr_values.append(psnr_value)
    tf.print(tf.reduce_mean(psnr_values))
