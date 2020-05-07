import kmeans as km
import numpy as np
import scipy.spatial.distance as ssdist


def _dunn_index(x, estimator):
    datasize = x.shape[0]
    intercluster_distance = ssdist.cdist(estimator.centroids, estimator.centroids, 'correlation')
    min_ctc = intercluster_distance[1, 0]
    for i in range(estimator.n_clusters):
        for j in range(estimator.n_clusters):
            if min_ctc > intercluster_distance[i, j] > 0:
                min_ctc = intercluster_distance[i, j]
    distance_ctp = ssdist.cdist(x, estimator.centroids, 'correlation')
    intracluster_distance = [distance_ctp[estimator.labels_[i]] for i in range(datasize)]
    max_intradist = np.max(intracluster_distance)
    return min_ctc / max_intradist


def choose_the_best(x, mincluster=2, maxcluster=10):
    the_best = km.KMeans(mincluster)
    estim = the_best.fit(x)
    best_dunn = _dunn_index(x, estim)
    for k in range(mincluster+1, maxcluster):
        estim_next = km.KMeans(k)
        estim_next = estim_next.fit(x)
        cur_dunn = _dunn_index(x, estim_next)
        if cur_dunn > best_dunn:
            best_dunn = cur_dunn
            estim = estim_next
    return estim
