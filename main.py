import os
import imageio

from sklearn.cluster import KMeans as ScikitKMeans
from clustering import KMeans as MyKMeans
from colors import get_unique_colors, replace_colors, round_colors, make_color_mean_map

K = 64

images_dir_in = "img_in"
images_dir_out = "img_out"
source_img_file = "apples.jpg"
# source_img_file = "road.jpg"
source_img_path = os.path.join(images_dir_in, source_img_file)

img = imageio.imread(source_img_path).astype(float)

unique_colors = list(get_unique_colors(img))
print("unique colors:", len(unique_colors))

# kmeans = MyKMeans(n_clusters=K)
kmeans = MyKMeans(n_clusters=K, init='kmeans++')
kmeans.fit(unique_colors, log=True)

# kmeans = ScikitKMeans(n_clusters=K, random_state=0)
# kmeans.fit(unique_colors)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

rounded_labels = round_colors(labels)
color_mean_map = make_color_mean_map(unique_colors, rounded_labels)

new_img = replace_colors(img, color_mean_map)

result_img_file = "apples_out_{}.png".format(K)
# result_img_file = "road_out_{}.png".format(K)
result_img_path = os.path.join(images_dir_out, result_img_file)

imageio.imwrite(result_img_path, new_img)
