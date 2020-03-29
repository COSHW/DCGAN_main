import IPython
import imageio
from PIL import Image
import glob
from IPython import display


def display_image(epoch_no):
    return Image.open('images/image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(50).show()


anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('images/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)
