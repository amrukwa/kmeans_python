import numpy as np
import initialization as init
import scipy.spatial.distance as ssdist


def initialization(n_clusters, initialize, x, random_state, metric):
    # this method initializes the first set of labels
    if initialize == 'random':
        centroids = init.random_init(x, n_clusters, random_state)
    elif initialize == 'k-means++':
        centroids = init.kpp_init(x, n_clusters, metric)
    return centroids


def labeling(centroids, x, metric):  # here sth doesn't work
    # this method creates labels for every position in training data based on current centroids
    distance = ssdist.cdist(x, centroids, metric)
    labels = distance.argmin(axis=1)
    return labels


def compute_centroids(n_clusters, centroids, labels, n_iter, max_iter, x, metric):
    while n_iter < max_iter:
        if n_iter > 0:
            prev_labels = labels
        centroids = _centroid_by_means(n_clusters, centroids, labels, x)
        labels = labeling(centroids, x, metric)
        if n_iter > 0 and np.all(labels == prev_labels):
            break
        n_iter += 1


def _centroid_by_means(n_clusters, centroids, labels, x):
    # this method takes already created labels to compute next centroids
    for k in range(n_clusters):
        centroids[k, :] = np.mean(x[labels == k, :], axis=0)
    return centroids


def inertia(centroids, x, metric):
    datasize = x.shape[0]
    distance = ssdist.cdist(x, centroids, metric)
    inertia = np.sum([(distance[i]) ** 2 for i in range(datasize)])
    return inertia


def subtract_mean(x):
    means_of_cols = np.mean(x, axis=0)
    feature_nr = x.shape[1]
    datasize = x.shape[0]
    substracted = np.array([[moving(x, i, j, means_of_cols) for i in range(feature_nr)] for j in range(datasize)])
    substracted.reshape((datasize, feature_nr))
    return substracted


def moving(x, dim1, dim2, means):
    value = x[dim2, dim1] - means[dim1]
    return value


def normalize(x):
    normalized = x
    subtracted_mean = subtract_mean(x)
    std_dev = np.std(x, axis=1)
    for i in range(x.shape[0]):
        if std_dev[i] == 0:
            print("Invalid data")
            exit(1)
        for j in range(x.shape[1]):
            normalized[i, j] = subtracted_mean[i, j]/std_dev[i]
    return normalized
