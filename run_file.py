import pandas as pd
import clusters_n as cn
import kmeans as km
import sklearn.datasets as sd
import sklearn.preprocessing as pre
import cross_validation as cv
import umap
import plotly.express as px
import sklearn.metrics as metrics


if __name__ == "__main__":
    iris = pd.read_csv("iris.data", header=None)
    labels_true = iris[4]
    data = iris.drop(columns=4).values
    # X, y = sd.make_blobs()
    transformer = pre.StandardScaler()
    data = transformer.fit_transform(data)
    est = km.KMeans(2)
    est = est.fit(data)
    dunn = cn.Dunn(est)
    est = dunn.fit(data)
    print(est.n_clusters)
    val = cv.KFold()
    score1 = val.validate(data, labels_true, est)
    print(score1)
