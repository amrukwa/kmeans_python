import numpy as np
import scipy.spatial.distance as ssdist


def initialization(n_clusters, initialize, x):
    # this method initializes the first set of labels
    datasize = x.shape[0]
    if initialize == 'random':
        np.random.seed(45)
        indices = np.random.choice(datasize, n_clusters, replace=False)
        centroids = x[indices]
    elif initialize == 'k-means++':
        first_cent = np.random.choice(datasize, 1)
        centroids = x[first_cent]
        k = 1
        while k < n_clusters:
            new_centre = _do_for_kpp(centroids, x)
            centroids = np.append(centroids, new_centre, axis=0)
            k += 1
    return centroids


def _do_for_kpp(centroids, x):
    datasize = x.shape[0]
    dist = fix_the_distance(x, centroids, 'correlation')
    p_dist = np.amin(dist, axis=1)
    all_dist = np.sum(p_dist)
    probability = [(p_dist[i]) / all_dist for i in range(datasize)]
    new_centre = x[np.random.choice(datasize, 1, p=probability)]
    return new_centre


def labeling(centroids, x):
    # this method creates labels for every position in training data based on current centroids
    distance = fix_the_distance(x, centroids, 'correlation')
    labels = np.argmin(distance, axis=1)
    return labels


def compute_centroids(n_clusters, centroids, labels, n_iter, max_iter, x):
    while n_iter < max_iter:
        if n_iter > 0:
            prev_centroids = centroids
        _centroid_by_means(n_clusters, centroids, labels, x)
        labels = labeling(centroids, x)
        if n_iter > 5 and np.all(centroids == prev_centroids):
            break
        n_iter += 1


def _centroid_by_means(n_clusters, centroids, labels, x):
    # this method takes already created labels to compute next centroids
    for k in range(n_clusters):
        centroids[k, :] = np.mean(x[labels == k, :], axis=0)


def inertia(centroids, x):
    datasize = x.shape[0]
    distance = fix_the_distance(x, centroids, 'correlation')
    inertia = np.sum([(distance[_]) ** 2 for _ in range(datasize)])
    return inertia


def fix_the_distance(x, centroids, dist_metric):
    distance = ssdist.cdist(x, centroids, metric=dist_metric)  # here is sth to change
    distance[np.isnan(distance)] = 0
    for i in range(x.shape[0]):
        for j in range(centroids.shape[0]):
            if (x[i] == centroids[j]).all():
                distance[i, j] = 0
    return distance
