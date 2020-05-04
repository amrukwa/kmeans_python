import pytest
from kmeans import KMeans
import numpy as np
import kmeans_functions as kmf
import sklearn.datasets as sd


def test_initializes():
    mdl = KMeans(3)
    assert mdl is not None


def test_fails_for_repeating_centroids():
    X, y = sd.make_blobs()
    for i in range(2, 11):
        centroids = kmf.initialization(i, 'k-means++', X)
        essence = np.unique(centroids, axis=0)
        assert essence.shape[0] == centroids.shape[0]


def test_fails_for_nonzero_distance():
    X, y = sd.make_blobs()
    distance = kmf.compute_distance(X, X, 'correlation')
    for i in range(X.shape[0]):
        assert distance[i, i] == 0
