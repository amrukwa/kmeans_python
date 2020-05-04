import kmeans as km
import numpy as np
import kmeans_functions as kmf


def _dunn_index(x, estimator):
    datasize = x.shape[0]
    intercluster_distance = kmf.compute_distance(estimator.centroids, estimator.centroids, 'correlation')
    min_ctc = intercluster_distance[1, 0]
    for i in range(estimator.n_clusters):
        for j in range(estimator.n_clusters):
            if min_ctc > intercluster_distance[i, j] > 0:
                min_ctc = intercluster_distance[i, j]
    distance_ctp = kmf.compute_distance(x, estimator.centroids, 'correlation')
    intracluster_distance = [distance_ctp[estimator.labels_[i]] for i in range(datasize)]
    max_intradist = np.max(intracluster_distance)
    return min_ctc / max_intradist


def choose_the_best(x, mincluster=2, maxcluster=10):
    the_best = km.KMeans(mincluster)
    estim = the_best.fit(x)
    best_index = _dunn_index(x, estim)
    estimators = [km.KMeans(mincluster)]
    estimators[0] = estimators[0].fit(x)
    best_index = 0
    best_dunn = _dunn_index(x, estimators[0])
    k = mincluster + 1
    ind = 1
    while k <= maxcluster:
        estimators.append(km.KMeans(k))
        estimators[ind] = estimators[ind].fit(x)
        cur_dunn = _dunn_index(x, estimators[ind])
        if cur_dunn > best_dunn:
            best_dunn = cur_dunn
            best_index = ind
        k += 1
        ind += 1
    return estimators[best_index]
