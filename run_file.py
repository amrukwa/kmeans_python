import pandas as pd
import clusters_n as cn
from sklearn.decomposition import PCA
import plotly.express as px
import umap
import sklearn.metrics as metrics
import numpy as np
import kmeans as km

if __name__ == "__main__":
    file = open('setA.txt')
    data = pd.read_csv('setA.txt', delimiter=' ')
    file.close()
    # data1 = data.drop(columns='Tissue').values
    data1 = data[["Gsto1", "Gstm1", "Cd9", "Prdx1", "Coro1a", "Rac2", "Perp"]].values
    estim = cn.choose_the_best(data1)
    labels = estim.labels_
    # labels = estim.fit_predict(data1)
    success_rate = metrics.adjusted_rand_score(data.Tissue.values, labels)
    print(success_rate)
    print(estim.n_clusters)


