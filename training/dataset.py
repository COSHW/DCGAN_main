import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import traceback
import matplotlib.pyplot as plt

def make_dataset(dataset_dir, img_size, colors, BUFFER_SIZE, BATCH_SIZE, progress_bar):
    train_images = []
    i = 1
    for item in os.listdir(dataset_dir):
        image = np.array(Image.open(os.path.join(dataset_dir, item)))
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

        train_images.append(image)
        if i == BUFFER_SIZE:
            break
        else:
            i += 1
        print(int(i/BUFFER_SIZE*100))
        # progress_bar.setProperty("value", int(i/BUFFER_SIZE)*100)

    train_images = np.array(train_images)
    # train_images = train_images.reshape(train_images.shape[0], img_size, img_size, colors).astype('float32')

    train_images = (train_images - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset

# make_dataset(r"D:\Projects\PythonProjects\4 kurs\PohProbuem\dataset", 32, 3, 10, 1, 1)
