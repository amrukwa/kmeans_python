import sklearn.base as base
import kmeans_functions as kme
import sklearn.utils as utils


class KMeans(base.ClusterMixin, base.BaseEstimator):

    def __init__(self,
                 n_clusters,
                 initialize='k-means++',
                 max_iter=1000,
                 metric='correlation',
                 random_state=None):
        self.n_clusters = n_clusters
        self.initialize = initialize
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state

    def fit(self, x):
        self.n_iter_ = 0
        self.random_state = utils.check_random_state(self.random_state)
        self.centroids = kme.initialization(self.n_clusters, self.initialize, x, self.random_state, self.metric)
        self.labels_ = kme.labeling(self.centroids, x, self.metric)
        kme.compute_centroids(self.n_clusters, self.centroids,
                              self.labels_, self.n_iter_, self.max_iter, x, self.metric)
        self.inertia = kme.inertia(self.centroids, x, self.metric)
        for i in range(9):
            n_iter = 0
            centroidsi = kme.initialization(self.n_clusters, self.initialize, x, self.random_state, self.metric)
            labelsi = kme.labeling(centroidsi, x, self.metric)
            kme.compute_centroids(self.n_clusters, centroidsi, labelsi, n_iter, self.max_iter, x, self.metric)
            inertia = kme.inertia(centroidsi, x, self.metric)
            if inertia < self.inertia:
                self.n_iter_ = n_iter
                self.inertia = inertia
                self.centroids = centroidsi
                self.labels_ = labelsi
        return self

    def predict(self, matrix):
        labels = kme.labeling(self.centroids, matrix, self.metric)
        return labels
