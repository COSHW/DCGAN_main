import matplotlib.pyplot as plt


def calc_size(length):
    if length == 1:
        return 1
    elif 1 < length < 5:
        return 2
    elif 4 < length < 10:
        return 3
    elif 9 < length <= 16:
        return 4


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(calc_size(predictions.shape[0]), calc_size(predictions.shape[0]), i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('images\image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
