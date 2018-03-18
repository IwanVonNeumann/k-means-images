import numpy as np


def get_unique_colors(img):
    n = img.shape[0]
    m = img.shape[1]

    pixels = img.reshape((n * m, 3))

    return {tuple(p) for p in pixels}


def round_color(color):
    return tuple([round(c) for c in color])


def round_colors(colors):
    return [round_color(c) for c in colors]


def replace_colors(img, color_mean_map):
    n = img.shape[0]
    m = img.shape[1]
    new_img = np.empty_like(img)

    for i in range(n):
        for j in range(m):
            point = tuple(img[i, j])
            new_color = color_mean_map[point]
            new_img[i, j] = new_color

    return new_img
