import numpy as np


def get_unique_colors(img):
    n, m, _ = img.shape
    pixels = img.reshape((n * m, 3))

    return {tuple(p) for p in pixels}


def round_color(color):
    return tuple([round(c) for c in color])


def round_colors(colors):
    return [round_color(c) for c in colors]


# TODO try vectorizing
def replace_colors(img, color_mean_map):
    new_img = np.empty_like(img, dtype=np.uint8)
    n, m, _ = new_img.shape

    for i in range(n):
        for j in range(m):
            point = tuple(img[i, j])
            new_color = color_mean_map[point]
            new_img[i, j] = new_color

    return new_img


def make_color_mean_map(colors, means):
    return {color: mean for color, mean in zip(colors, means)}
