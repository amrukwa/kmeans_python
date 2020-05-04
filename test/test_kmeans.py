import pytest
from kmeans import KMeans
import numpy as np
import kmeans_functions as kmf
import sklearn.datasets as sd


def test_initializes():
    mdl = KMeans(3)
    assert mdl is not None


@pytest.mark.parametrize("test_input,expected", [(i, 0) for i in range(100)])
def test_fails_for_nonzero_distance(test_input, expected):
    X, y = sd.make_blobs()
    distance = kmf.compute_distance(X, X, 'correlation')
    assert distance[test_input, test_input] == expected


@pytest.mark.parametrize("test_input,expected", [(i, i) for i in range(2, 11)])
def test_fails_for_repeating_centroids(test_input, expected):
    X, y = sd.make_blobs()
    centroids = kmf.initialization(test_input, 'k-means++', X)
    essence = np.unique(centroids, axis=0)
    assert essence.shape[0] == expected
