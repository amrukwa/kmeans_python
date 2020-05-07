import kmeans as km
import numpy as np
import scipy.spatial.distance as ssdist
import sklearn.preprocessing as preprocessing
import kmeans_functions as kmf


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


def choose_the_best(x, mincluster=2, maxcluster=10, metric='correlation'):
    if metric == 'correlation':
        transformer = preprocessing.StandardScaler()
        normalized_data = transformer.fit_transform(x)
        # normalized_data = kmf.normalize(x)
    else:
        normalized_data = x
    the_best = km.KMeans(mincluster,
                         metric=metric)
    estim = the_best.fit(normalized_data)
    best_dunn = _dunn_index(normalized_data, estim, metric)
    for k in range(mincluster+1, maxcluster):
        estim_next = km.KMeans(k,
                               metric=metric)
        estim_next = estim_next.fit(normalized_data)
        cur_dunn = _dunn_index(normalized_data, estim_next, metric)
        if cur_dunn > best_dunn:
            best_dunn = cur_dunn
            estim = estim_next
    return estim
