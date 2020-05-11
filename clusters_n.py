import kmeans as km
import numpy as np
import scipy.spatial.distance as ssdist


class Dunn(km.KMeans):
    def __init__(self,
                 estimator,
                 mincluster=2,
                 maxcluster=10):
        self.estimator = estimator
        self.mincluster = mincluster
        self.maxcluster = maxcluster

    def fit(self, x):
        self.dunn = _dunn_index(x, self.estimator, self.estimator.metric)
        for k in range(self.mincluster + 1, self.maxcluster):
            estim_next = km.KMeans(k,
                                   metric=self.estimator.metric)
            estim_next = estim_next.fit(x)
            cur_dunn = _dunn_index(x, estim_next, self.estimator.metric)
            if cur_dunn > self.dunn:
                self.dunn = cur_dunn
                self.estimator = estim_next
        return self.estimator


def _dunn_index(x, estimator, metric):
    datasize = x.shape[0]
    intercluster_distance = ssdist.cdist(estimator.centroids, estimator.centroids, metric)
    min_ctc = intercluster_distance[1, 0]
    for i in range(estimator.n_clusters):
        for j in range(estimator.n_clusters):
            if min_ctc > intercluster_distance[i, j] > 0:
                min_ctc = intercluster_distance[i, j]
    distance_ctp = ssdist.cdist(x, estimator.centroids, metric)
    intracluster_distance = [distance_ctp[estimator.labels_[i]] for i in range(datasize)]
    max_intradist = np.max(intracluster_distance)
    return min_ctc / max_intradist
