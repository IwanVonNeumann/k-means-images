import math
import random
import numpy

from colors import round_color


def kmeans(points, k, log=False):
    previous_err = math.inf
    delta_err = math.inf
    i = 1

    means = random_means(points, k)

    while delta_err > 0:
        if log:
            print("iteration {}".format(i))

        clusters = assign_clusters(points, means)
        grouped_points = group_points(points, clusters)
        means = recalculate_means(grouped_points)

        current_err = cluster_error_sum(points, clusters)
        delta_err = previous_err - current_err
        previous_err = current_err

        if log:
            print("total error: {:.3f}".format(current_err))
            print("delta error: {:.3f}".format(delta_err))
            i += 1

    return means


def random_means(items, k):
    return random.sample(items, k)


def assign_clusters(points, means):
    return [nearest_mean(p, means) for p in points]


def nearest_mean(point, means):
    nearest = None
    min_dist = math.inf
    for m in means:
        d = distance(point, m)
        if d < min_dist:
            nearest = m
            min_dist = d
    return nearest


def distance(X, Y):
    if len(X) != len(Y):
        raise ValueError("Dimensions of objects do not match")

    return sum([(float(x) - float(y)) ** 2 for x, y in zip(X, Y)])


def group_points(points, cluster_labels):
    clusters = set(cluster_labels)
    cp_pairs = zip(cluster_labels, points)
    return {cluster: [p for c, p in cp_pairs if c == cluster] for cluster in clusters}


def recalculate_means(grouped_points):
    return [calculate_mean(points) for points in grouped_points.values()]


def calculate_mean(points):
    return tuple(map(numpy.mean, zip(*points)))


def cluster_error_sum(points, clusters):
    grouped_points = group_points(points, clusters)
    return sum([cluster_error(points_group, mean) for mean, points_group in grouped_points.items()])


def cluster_error(points, mean):
    return sum([distance(p, mean) for p in points])


def make_color_mean_map(colors, means):
    return {color: round_color(mean) for color, mean in zip(colors, means)}
