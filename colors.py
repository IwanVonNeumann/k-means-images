import numpy as np


def get_unique_colors(img):
    n = img.shape[0]
    m = img.shape[1]
    points = set()

    for i in range(n):
        for j in range(m):
            point = (img[i, j, 0], img[i, j, 1], img[i, j, 2])
            points.add(point)

    return points


def round_color(color):
    return tuple([round(color[0]), round(color[1]), round(color[2])])


def replace_colors(img, color_mean_map):
    n = img.shape[0]
    m = img.shape[1]
    new_img = np.copy(img)

    for i in range(n):
        for j in range(m):
            point = (img[i, j, 0], img[i, j, 1], img[i, j, 2])
            new_color = color_mean_map[point]
            for c in range(3):
                new_img[i, j, c] = new_color[c]

    return new_img
