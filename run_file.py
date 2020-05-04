import pandas as pd
import clusters_n as cn
import sklearn.datasets as sd

if __name__ == "__main__":
    X, y = sd.make_blobs()
    estim = cn.choose_the_best(X)
