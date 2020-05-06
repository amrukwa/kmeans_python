import pytest
from kmeans import KMeans
import numpy as np
import kmeans_functions as kmf
import sklearn.datasets as sd


def test_initializes():
    mdl = KMeans(3)
    assert mdl is not None


@pytest.mark.parametrize("test_input,expected", [(i, i) for i in range(2, 11)])
def test_fails_for_repeating_centroids(test_input, expected):
    X, y = sd.make_blobs()
    centroids = kmf.initialization(test_input, 'k-means++', X, random_state=None, metric='correlation')
    essence = np.unique(centroids, axis=0)
    assert essence.shape[0] == expected
