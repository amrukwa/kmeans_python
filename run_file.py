import pandas as pd
import clusters_n as cn
import kmeans as km
import sklearn.datasets as sd
import sklearn.preprocessing as pre
import cross_validation as cv
import umap
import plotly.express as px
import numpy as np
import kmedoids as kmd

if __name__ == "__main__":
    iris = pd.read_csv("iris.data", header=None)
    labels_true = iris[4]
    data = iris.drop(columns=4).values
    transformer = pre.StandardScaler()
    data1 = transformer.fit_transform(data)
    est = kmd.KMedoids(2)
    est = est.fit(data)
    val1 = cv.KFold()
    score1 = val1.validate(data, labels_true, est)
    print(score1)

