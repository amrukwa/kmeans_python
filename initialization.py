import numpy as np
import kmeans_functions as kmf


def random_init(data, n_clusters, random_state):
    indices = np.random.choice(data.shape[0], n_clusters, replace=False, seed=random_state)
    centroids = data[indices]
    return centroids


def kpp_init(data, n_clusters, metric):
    datasize = data.shape[0]
    first_cent = np.random.choice(datasize, 1)
    centroids = data[first_cent]
    k = 1
    while k < n_clusters:
        new_centre = next_for_kpp(centroids, data, metric)
        centroids = np.append(centroids, new_centre, axis=0)
        k += 1
    return centroids


def next_for_kpp(centroids, x, metric):
    datasize = x.shape[0]
    dist = kmf.compute_distance(x, centroids, metric)
    p_dist = dist.min(axis=1)
    p_dist = p_dist**2
    all_dist = np.sum(p_dist)
    probability = [(p_dist[i]) / all_dist for i in range(datasize)]
    new_centre = x[np.random.choice(datasize, 1, p=probability)]
    return new_centre
