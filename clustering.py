import math
import random
import numpy as np

from scipy.spatial.distance import cdist

from time_utils import measure_time


@measure_time
def kmeans(points, k, log=False):
    previous_err = math.inf
    delta_err = math.inf
    i = 1

    means = random_means(points, k)

    while delta_err > 0:
        if log:
            print("iteration {}".format(i))

        cluster_labels = assign_labels(points, means)
        grouped_points = group_points(points, cluster_labels)
        means = recalculate_means(grouped_points)

        current_err = cluster_error_sum(points, cluster_labels)
        delta_err = previous_err - current_err
        previous_err = current_err

        if log:
            print("total error: {:.3f}".format(current_err))
            print("delta error: {:.3f}".format(delta_err))
            i += 1

    return means


def random_means(items, k):
    return random.sample(items, k)


def assign_labels(points, means):
    D = cdist(np.array(points), np.array(means), metric='sqeuclidean')
    nearest_means_indices = np.argmin(D, axis=1)
    return [means[i] for i in nearest_means_indices]


# TODO think of numpy
def group_points(points, cluster_labels):
    clusters = set(cluster_labels)
    return {cluster: [p for c, p in zip(cluster_labels, points) if c == cluster] for cluster in clusters}


def recalculate_means(grouped_points):
    return [mean_of_cluster(points) for points in grouped_points.values()]


def mean_of_cluster(points):
    return tuple(np.array(points).mean(axis=0))


# TODO think of numpy
def cluster_error_sum(points, cluster_labels):
    grouped_points = group_points(points, cluster_labels)
    return sum([cluster_error(points_group, mean) for mean, points_group in grouped_points.items()])


def cluster_error(points, mean):
    return np.sum(cdist(np.array(points), np.array([mean]), metric='sqeuclidean'))


def make_color_mean_map(colors, means):
    return {color: mean for color, mean in zip(colors, means)}
