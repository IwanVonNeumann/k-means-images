import pylab as pl

from os import listdir
from os.path import isfile, join

IMAGES_SERIES_DIR = "img_series"

all_file_names = [file_name for file_name in listdir(IMAGES_SERIES_DIR) if isfile(join(IMAGES_SERIES_DIR, file_name))]
all_file_paths = [join(IMAGES_SERIES_DIR, f) for f in all_file_names]

while True:
    i = 0
    img = None

    for f in all_file_paths:
        im = pl.imread(f)

        if img is None:
            img = pl.imshow(im)
        else:
            img.set_data(im)

        pl.title('iteration {}'.format(i))
        pl.pause(0.05)
        pl.draw()
        i += 1

    pl.pause(3)
