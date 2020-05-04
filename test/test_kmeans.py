from kmeans import KMeans
import numpy as np
import kmeans_functions as kmf
import pandas as pd
import initialization as init


def test_initializes():
    mdl = KMeans(3)
    assert mdl is not None


def test_fails_for_repeating_centroids():
    file = open('/Users/amruk/PycharmProjects/kmeans/setA.txt')
    data = pd.read_csv('/Users/amruk/PycharmProjects/kmeans/setA.txt', delimiter=' ')
    file.close()
    data1 = data[["Gsto1", "Gstm1", "Cd9", "Prdx1", "Coro1a", "Rac2", "Perp"]].values
    for i in range(2, 11):
        centroids = kmf.initialization(i, 'k-means++', data1)
        essence = np.unique(centroids, axis=0)
        assert essence.shape[0] == centroids.shape[0]


def test_fails_for_nonzero_distance():
    file = open('/Users/amruk/PycharmProjects/kmeans/setA.txt')
    data = pd.read_csv('/Users/amruk/PycharmProjects/kmeans/setA.txt', delimiter=' ')
    file.close()
    data1 = data[["Gsto1", "Gstm1", "Cd9", "Prdx1", "Coro1a", "Rac2", "Perp"]].values
    distance = init.fix_the_distance(data1, data1, 'correlation')
    for i in range(data1.shape[0]):
        assert distance[i, i] == 0
