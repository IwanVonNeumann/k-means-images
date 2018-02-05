import os
import imageio

from sklearn.cluster import KMeans
from clustering import make_color_mean_map, kmeans, assign_clusters
from colors import get_unique_colors, replace_colors

K = 64

images_dir_in = "img_in"
images_dir_out = "img_out"

source_img_file = "apples.jpg"

source_img_path = os.path.join(images_dir_in, source_img_file)
img = imageio.imread(source_img_path)

unique_colors = list(get_unique_colors(img))
print("unique colors:", len(unique_colors))

# centers = kmeans(unique_colors, K, log=True) # manual implementation
centers = KMeans(n_clusters=K, random_state=0).fit(unique_colors).cluster_centers_  # scikit-learn

final_clusters = assign_clusters(unique_colors, centers)
color_mean_map = make_color_mean_map(unique_colors, final_clusters)

new_img = replace_colors(img, color_mean_map)

result_img_file = "apples_out_{}.png".format(K)

result_img_path = os.path.join(images_dir_out, result_img_file)
imageio.imwrite(result_img_path, new_img)
