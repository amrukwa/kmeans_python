import kmeans as km
import kmeans_functions as kmf
import sklearn.utils as utils
import scipy.spatial.distance as ssdist
import numpy as np


class KMedoids(km.KMeans):
    def fit(self, x):
        self.distance = ssdist.cdist(x, x, self.metric)
        self.n_iter_ = 0
        self.random_state = utils.check_random_state(self.random_state)
        self.centroids_indices_ = initialization(self.n_clusters, self.initialize, x, self.random_state, self.metric)
        self.labels_ = labeling(self.centroids_indices_, self.distance)
        compute_indices(self.centroids_indices_, self.labels_, self.n_iter_, self.max_iter, self.distance)
        self.inertia = kmf.inertia(x[self.centroids_indices_], x, self.metric)  # refactor
        for i in range(9):
            n_iter = 0
            centroids_ind = initialization(self.n_clusters, self.initialize, x, self.random_state, self.metric)
            labelsi = labeling(self.centroids_indices_, self.distance)
            compute_indices(centroids_ind, labelsi, n_iter, self.max_iter, self.distance)
            inertia = kmf.inertia(x[centroids_ind], x, self.metric)  # refactor
            if inertia < self.inertia:
                self.n_iter_ = n_iter
                self.inertia = inertia
                self.centroids_indices_ = centroids_ind
                self.labels_ = labelsi
        self.centroids = x[self.centroids_indices_]
        return self

    def predict(self, matrix):
        labels = kmf.labeling(self.centroids, matrix, self.metric)
        return labels


def initialization(n_clusters, initialize, x, random_state, metric):
    # this method initializes the first set of labels
    if initialize == 'random':
        centroids_indices = random_init(x, n_clusters, random_state)
    elif initialize == 'k-means++':
        centroids_indices = kpp_init(x, n_clusters, metric)
    return centroids_indices


def random_init(data, n_clusters, random_state):
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    return indices


def next_for_kpp(centroids_indices, x, metric):
    datasize = x.shape[0]
    dist = ssdist.cdist(x, x[centroids_indices], metric)  # change for better performance
    p_dist = dist.min(axis=1)
    p_dist = p_dist**2
    all_dist = np.sum(p_dist)
    probability = [(p_dist[i]) / all_dist for i in range(datasize)]
    new_index = np.random.choice(datasize, 1, p=probability)
    return new_index


def kpp_init(data, n_clusters, metric):
    datasize = data.shape[0]
    centroid_indices = np.array(np.random.choice(datasize, 1))
    k = 1
    while k < n_clusters:
        new_centre = next_for_kpp(centroid_indices, data, metric)
        centroids_indices = np.append(centroid_indices, new_centre, axis=0)
        k += 1
    return centroids_indices


def labeling(centroids_indices, distance):
    dist_local = distance[:, centroids_indices]
    labels = dist_local.argmin(axis=1)
    return labels


def indices_by_distance(distance, centroids_indices, labels):
    for i in range(len(centroids_indices)):
        sum_dist = np.sum(distance[labels == i], axis=1)
        new_index = sum_dist.argmin()
        centroids_indices[i] = new_index
    return centroids_indices


def compute_indices(centroids_indices, labels, n_iter, max_iter, distance):
    while n_iter < max_iter:
        if n_iter > 0:
            prev_indices = centroids_indices
        centroids_indices = indices_by_distance(distance, centroids_indices, labels)
        labels = labeling(centroids_indices, distance)
        if n_iter > 0 and np.all(centroids_indices == prev_indices):
            break
        n_iter += 1
