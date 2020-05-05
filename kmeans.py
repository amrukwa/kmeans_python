import sklearn.base as base
import kmeans_functions as kme


class KMeans(base.ClusterMixin, base.BaseEstimator):
    def __init__(self, n_clusters, initialize='k-means++', max_iter=1000, random_state=None):
        self.n_clusters = n_clusters
        self.initialize = initialize
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x):
        self.n_iter_ = 0
        self.random_state = kme.check_random_state(self.random_state)
        self.centroids = kme.initialization(self.n_clusters, self.initialize, x, self.random_state)
        self.labels_ = kme.labeling(self.centroids, x)
        kme.compute_centroids(self.n_clusters, self.centroids, self.labels_, self.n_iter_, self.max_iter, x)
        self.inertia = kme.inertia(self.centroids, x)
        for i in range(9):
            n_iter = 0
            centroidsi = kme.initialization(self.n_clusters, self.initialize, x, self.random_state)
            labelsi = kme.labeling(centroidsi, x)
            kme.compute_centroids(self.n_clusters, centroidsi, labelsi, n_iter, self.max_iter, x)
            inertia = kme.inertia(centroidsi, x)
            if inertia < self.inertia:
                self.n_iter_ = n_iter
                self.inertia = inertia
                self.centroids = centroidsi
                self.labels = labelsi
        return self
