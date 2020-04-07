import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2


def make_dataset(dataset_dir, img_size, BUFFER_SIZE, BATCH_SIZE):
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
        print(int(i/BUFFER_SIZE*100)) if i % 1000 == 0 else 1

    train_images = np.array(train_images, dtype='float32')
    for i in range(len(train_images)):
        train_images[i] = (train_images[i] - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset
