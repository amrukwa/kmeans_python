import numpy as np
import kmeans_functions as kmf


def random_init(data, n_clusters):
    np.random.seed(45)
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    centroids = data[indices]
    return centroids


def kpp_init(data, n_clusters):
    datasize = data.shape[0]
    first_cent = np.random.choice(datasize, 1)
    centroids = data[first_cent]
    k = 1
    while k < n_clusters:
        new_centre = _do_for_kpp(centroids, data)
        centroids = np.append(centroids, new_centre, axis=0)
        k += 1
    return centroids


def _do_for_kpp(centroids, x):
    datasize = x.shape[0]
    dist = kmf.fix_the_distance(x, centroids, 'correlation')
    p_dist = np.amin(dist, axis=1)
    all_dist = np.sum(p_dist)
    probability = [(p_dist[i]) / all_dist for i in range(datasize)]
    new_centre = x[np.random.choice(datasize, 1, p=probability)]
    return new_centre
