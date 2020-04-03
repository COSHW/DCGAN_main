import matplotlib.pyplot as plt
import numpy as np


def calc_size(length):
    if length == 1:
        return 1
    elif 1 < length < 5:
        return 2
    elif 4 < length < 10:
        return 3
    elif 9 < length <= 16:
        return 4


def generate_and_save_images(model, epoch, test_input, colors):
    fig = plt.figure(figsize=(4, 4))
    predictions = model(test_input, training=False)
    for i in range(predictions.shape[0]):
        img = np.array(predictions[i, :, :] * 127.5 + 127.5, np.int32)
        plt.subplot(calc_size(predictions.shape[0]), calc_size(predictions.shape[0]), i + 1)
        plt.imshow(img, cmap='gray' if colors == 1 else None)

        plt.axis('off')

    plt.savefig('images\{}.png'.format(epoch))
    plt.close()
