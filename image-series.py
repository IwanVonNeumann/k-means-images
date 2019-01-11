import os

import imageio

from clustering import *
from colors import get_unique_colors, replace_colors, round_colors

IMAGES_DIR_IN = "img_in"
IMAGES_DIR_OUT = "img_out"
SERIES_DIR = 'img_series'
# SOURCE_IMAGE_FILE_NAME = "road.jpg"
SOURCE_IMAGE_FILE_NAME = "apples.jpg"

source_img_file_path = os.path.join(IMAGES_DIR_IN, SOURCE_IMAGE_FILE_NAME)

img_in = imageio.imread(source_img_file_path).astype(float)

unique_colors = list(get_unique_colors(img_in))
print("unique colors:", len(unique_colors))

K = 16

previous_err = math.inf
delta_err = math.inf

points = unique_colors
means = random_means(points, K)

labels = assign_labels(unique_colors, means)
rounded_labels = round_colors(labels)
color_mean_map = make_color_mean_map(unique_colors, rounded_labels)

img_out = replace_colors(img_in, color_mean_map)

result_img_file = "apples_f_{:03d}.png".format(0)
result_img_path = os.path.join(SERIES_DIR, result_img_file)

imageio.imwrite(result_img_path, img_out)

log = True
i = 1

while delta_err > 0:
    if log:
        print("iteration {}".format(i))

    cluster_labels = assign_labels(points, means)
    grouped_points = group_points(points, cluster_labels)
    means = recalculate_means(grouped_points)

    current_err = cluster_error_sum(points, cluster_labels)
    delta_err = previous_err - current_err
    previous_err = current_err

    labels = assign_labels(unique_colors, means)
    rounded_labels = round_colors(labels)
    color_mean_map = make_color_mean_map(unique_colors, rounded_labels)

    img_out = replace_colors(img_in, color_mean_map)

    result_img_file = "apples_f_{:03d}.png".format(i)
    result_img_path = os.path.join(SERIES_DIR, result_img_file)

    imageio.imwrite(result_img_path, img_out)

    if log:
        print("total error: {:.3f}".format(current_err))
        print("delta error: {:.3f}".format(delta_err))
        print('written to', result_img_file)
        i += 1
