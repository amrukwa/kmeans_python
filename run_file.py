import pandas as pd
import clusters_n as cn
import kmeans as km
import sklearn.datasets as sd
import sklearn.preprocessing as preprocessing
import umap
import plotly.express as px


if __name__ == "__main__":
    iris = pd.read_csv("iris.data", header=None)
    data = iris.drop(columns=4).values
    # X, y = sd.make_blobs()
    transformer = preprocessing.StandardScaler()
    normalized_data = transformer.fit_transform(data)
    estim = km.KMeans(2)
    estim = estim.fit(data)
    dunn = cn.Dunn(estim)
    estim = dunn.fit(data)
    print(estim.n_clusters)
    print(estim.labels)

