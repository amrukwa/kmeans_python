import numpy as np
import initialization as init
import scipy.spatial.distance as ssdist


def initialization(n_clusters, initialize, x):
    # this method initializes the first set of labels
    if initialize == 'random':
        centroids = init.random_init(x, n_clusters)
    elif initialize == 'k-means++':
        centroids = init.kpp_init(x, n_clusters)
    return centroids


def labeling(centroids, x):
    # this method creates labels for every position in training data based on current centroids
    distance = compute_distance(x, centroids, 'correlation')
    labels = np.argmin(distance, axis=1)
    return labels


def compute_centroids(n_clusters, centroids, labels, n_iter, max_iter, x):
    while n_iter < max_iter:
        if n_iter > 0:
            prev_centroids = centroids
        centroids = _centroid_by_means(n_clusters, centroids, labels, x)
        labels = labeling(centroids, x)
        if n_iter > 5 and np.all(centroids == prev_centroids):
            break
        n_iter += 1


def _centroid_by_means(n_clusters, centroids, labels, x):
    # this method takes already created labels to compute next centroids
    for k in range(n_clusters):
        centroids[k, :] = np.mean(x[labels == k, :], axis=0)
    return centroids


def inertia(centroids, x):
    datasize = x.shape[0]
    distance = compute_distance(x, centroids, 'correlation')
    inertia = np.sum([(distance[_]) ** 2 for _ in range(datasize)])
    return inertia


def compute_distance(x, centroids, dist_metric):
    distance = ssdist.cdist(x, centroids, metric=dist_metric)  # here is sth to change
    distance[np.isnan(distance)] = 0
    for i in range(x.shape[0]):
        for j in range(centroids.shape[0]):
            if (x[i] == centroids[j]).all():
                distance[i, j] = 0
    return distance
