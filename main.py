from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from training import models, dataset, train, generate_image
import random
import numpy
import os
import traceback

async def pb(pp, v):
    pp.setValue(v)
# создание новой модели или тренинг существующей
async def create_model(dataset_dir, img_size, colors, buffer_size, batch_size, epochs, num_examples_to_generate, exist_model_file, time, label_top, new_modal_file, progress):
    # проверка на основные данные
    if new_modal_file != "":
        if new_modal_file[-3:] != '.h5':
            new_modal_file = new_modal_file + '.h5'
    else:
        new_modal_file = 'new_modal.h5'

    if epochs == "":
        if time == "":
            return "Error_Need_Time_Or_Epochs"

    if dataset_dir == "":
        return "Error_Dataset_Dir"

    if buffer_size == "":
        buffer_size = len(os.listdir(dataset_dir))
        if buffer_size > 20000:
            buffer_size = 20000

    # <------------------------------------------------------------------------------------------------->

    # создание seed
    seed = []
    for i in range(int(num_examples_to_generate)):
        l = []
        for j in range(100):
            num = round(random.uniform(-1, 1), 8)
            l.append(num)
        seed.append(l)
    seed = numpy.array(seed)

    # <------------------------------------------------------------------------------------------------->

    # создание тренировочных данных
    train_dataset = dataset.make_dataset(dataset_dir, int(img_size), int(buffer_size), int(batch_size))
    await pb(progress, 20)
    # <------------------------------------------------------------------------------------------------->

    # создание модели в зависимости от размера изображения
    model_maker = models.ModelMaking()
    # генератор
    if exist_model_file == "":
        if int(img_size) == 64:
            generator = model_maker.make_generator_model_64(int(batch_size), int(img_size), int(colors))
        elif int(img_size) == 128:
            generator = model_maker.make_generator_model_128(int(batch_size), int(img_size), int(colors))
        elif int(img_size) == 256:
            generator = model_maker.make_generator_model_256(int(batch_size), int(img_size), int(colors))
        elif int(img_size) == 28:
            generator = model_maker.make_generator_model_28(int(batch_size), int(img_size), int(colors))
        else:
            generator = model_maker.make_generator_model_32_test(int(batch_size), int(img_size), int(colors))
    else:
        # загрузка внешней модели, если существует
        try:
            if exist_model_file[-3:] != '.h5':
                exist_model_file = exist_model_file + '.h5'
            generator = tf.keras.models.load_model(exist_model_file)
        except:
            traceback.print_exc()
            return "Error_Model_Not_Exist"
    # дискриминатор
    discriminator = model_maker.make_discriminator_model(int(batch_size), int(img_size), int(colors))

    # <------------------------------------------------------------------------------------------------->

    # запуск тренировки
    trainer = train.Training(batch_size=int(batch_size), generator=generator,
                                         discriminator=discriminator, seed=seed, colors=int(colors))

    trainer.train_start(train_dataset, epochs, time, new_modal_file)

    label_top.setText("Готово!")


# генерация изображения
def create(model_file, image_name):
    seed = []
    for i in range(int(9)):
        l = []
        for j in range(100):
            num = round(random.uniform(-1, 1), 8)
            l.append(num)
        seed.append(l)
    seed = numpy.array(seed)
    generator = tf.keras.models.load_model(model_file, compile=False)
    generate_image.generate_and_save_images(generator, image_name, seed, manual=True)


