import pandas as pd
import clusters_n as cn
import kmeans as km
import sklearn.preprocessing as pre
import cross_validation as cv
import umap
import plotly.express as px
import kmedoids as kmd

if __name__ == "__main__":
    iris = pd.read_csv("iris.data", header=None)
    labels_true = iris[4]
    data = iris.drop(columns=4).values
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    est = kmd.KMedoids(2)
    est = est.fit(data)
    labels = est.labels_
    val1 = cv.KFold()
    score1 = val1.validate(data, labels_true, est)
    print(score1)
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    df['Label'] = labels
    f = px.scatter(df, x="x", y="y", color="Label", marginal_y="histogram", marginal_x="histogram")
    f.write_html("kmedoids.html")

    transformer = pre.StandardScaler()
    data1 = transformer.fit_transform(data)
