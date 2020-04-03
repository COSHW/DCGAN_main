from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from training import models, dataset, train, generate_image
import random
import numpy
import os
import traceback


def create_model(dataset_dir, img_size, colors, buffer_size, batch_size, epochs, num_examples_to_generate, model_file, time, progress_bar, label_top, label_low):
    if epochs == "":
        if time == "":
            return "Error_Need_Time_Or_Epochs"
    if dataset_dir == "":
        return "Error_Dataset_Dir"
    print("ch1")
    if buffer_size == "":
        buffer_size = len(os.listdir(dataset_dir))
        if buffer_size > 20000:
            buffer_size = 20000

    noise_dim = 100
    seed = []
    for i in range(int(num_examples_to_generate)):
        l = []
        for j in range(noise_dim):
            num = round(random.uniform(-1, 1), 8)
            l.append(num)
        seed.append(l)
    seed = numpy.array(seed)
    print("ch2")
    train_dataset = dataset.make_dataset(dataset_dir, int(img_size), int(colors), int(buffer_size), int(batch_size), progress_bar)
    print("ch3")
    try:
        progress_bar.setProperty("value", 0)
        model_maker = models.ModelMaking()
        if model_file == "":
            if int(img_size) == 128:
                generator = model_maker.make_generator_model_128(int(batch_size), int(img_size), int(colors))
            elif int(img_size) == 256:
                generator = model_maker.make_generator_model_256(int(batch_size), int(img_size), int(colors))
            else:
                generator = model_maker.make_generator_model_28_32_64(int(batch_size), int(img_size), int(colors))
        else:
            try:
                generator = tf.keras.models.load_model(model_file)
            except:
                return "Error_Model_Not_Exist"
        if int(img_size) == 256:
            discriminator = model_maker.make_discriminator_model_256(int(batch_size), int(img_size), int(colors))
        else:
            discriminator = model_maker.make_discriminator_model(int(batch_size), int(img_size), int(colors))
    except:
        traceback.print_exc()
    try:
        trainer = train.Training(batch_size=int(batch_size), noise_dim=noise_dim, generator=generator,
                                         discriminator=discriminator, seed=seed, colors=int(colors))
    except:
        traceback.print_exc()
    label_top.setText("Начал обучение")
    try:
        trainer.train_start(train_dataset, epochs, time)
    except:
        traceback.print_exc()
    label_top.setText("Готово!")


def create():
    seed = []
    for i in range(int(9)):
        l = []
        for j in range(100):
            num = round(random.uniform(-1, 1), 8)
            l.append(num)
        seed.append(l)
    seed = numpy.array(seed)

    generator = tf.keras.models.load_model(r"model.h5")
    generate_image.generate_and_save_images(generator, 1, seed, 3)

