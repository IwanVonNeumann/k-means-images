import math
import random
import numpy as np

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
    return [nearest_mean(p, means) for p in points]


def nearest_mean(point, means):
    return min(means, key=lambda mean: distance(mean, point))


def distance(X, Y):
    return sum([(x - y) ** 2 for x, y in zip(X, Y)])


def group_points(points, cluster_labels):
    clusters = set(cluster_labels)
    return {cluster: [p for c, p in zip(cluster_labels, points) if c == cluster] for cluster in clusters}


def recalculate_means(grouped_points):
    return [mean_of_cluster(points) for points in grouped_points.values()]


def mean_of_cluster(points):
    return tuple(map(np.mean, zip(*points)))


def cluster_error_sum(points, cluster_labels):
    grouped_points = group_points(points, cluster_labels)
    return sum([cluster_error(points_group, mean) for mean, points_group in grouped_points.items()])


def cluster_error(points, mean):
    return sum([distance(p, mean) for p in points])


def make_color_mean_map(colors, means):
    return {color: mean for color, mean in zip(colors, means)}
