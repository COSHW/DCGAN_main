import IPython
import imageio
from PIL import Image
import glob
from IPython import display
import os

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = os.listdir('static/Новая папка')
    files = []
    for item in filenames:
        item = item.replace(".png", "")
        files.append(int(item))
    files = sorted(files)
    for i, filename in enumerate(files):
        image = imageio.imread(os.path.join('static/Новая папка', str(filename)+'.png'))
        writer.append_data(image)
