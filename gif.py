import imageio
import os


def gif_gen():
    anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = os.listdir('training_images')
        files = []
        for item in filenames:
            item = item.replace(".png", "")
            files.append(int(item))
        files = sorted(files)
        for i, filename in enumerate(files):
            image = imageio.imread(os.path.join('training_images', str(filename)+'.png'))
            writer.append_data(image)
