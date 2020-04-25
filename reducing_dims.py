import numpy as np
from scipy.linalg import orth
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


class PCA:
    def __init__(self, x, reduce_to2="NO", reduced_dims=2):
        self.x = x
        self.reduced_dims = reduced_dims
        self.reduce_to2 = reduce_to2
        self.feature_nr = x.shape[1]
        self.datasize = x.shape[0]

    def pca(self):
        self._subtract_mean()
        cov_moved = np.cov(np.transpose(self.x))
        self.evecs = orth(cov_moved)
        for i in range(self.feature_nr):
            self._power_method(cov_moved, i)
        self.evals = np.array([self._rayleigh_quotient(cov_moved, i) for i in range(self.feature_nr)])
        importance = self.evals.argsort()[::-1]
        self.evals = self.evals[importance]
        self.evecs = self.evecs[importance]
        self.x = self.dot_product(self.x, self.evecs)
        if self.reduce_to2 == "NO":
            self._choose_dims()
        self.x = self.x[:, :self.reduced_dims]
        # for_visualization = pd.DataFrame(self.x, columns=['x', 'y'])
        return self.x  # reduced dim data

    def _subtract_mean(self):
        means_of_cols = np.mean(self.x, axis=0)
        self.x = np.array([[self._moving(i, j, means_of_cols) for i in range(self.feature_nr)] for j in range(self.datasize)])
        self.x.reshape((self.datasize, self.feature_nr))

    def _moving(self, dim1, dim2, means):
        value = self.x[dim2, dim1] - means[dim1]
        return value

    def _power_method(self, matrix, i, tol=0.0001):
        iters = 0
        diff = np.array([100 for _ in range(matrix.shape[1])])
        while iters < 50 and diff.all() > tol:
            prev_einv = self.evecs[i]
            self.evecs[i] = self.dot_product(matrix, prev_einv) / np.linalg.norm(self.dot_product(matrix, prev_einv))
            diff = np.absolute(self.evecs[i] - prev_einv)
            iters += 1

    def _rayleigh_quotient(self, matrix, i):
        eigenvalue = (self.dot_product(matrix, self.evecs[i]))
        eigenvalue = self.dot_product(self.evecs[i], eigenvalue) / self.dot_product(self.evecs[i], self.evecs[i])
        return eigenvalue

    def _choose_dims(self):
        sum = np.sum(self.evals)
        dim_number = 0
        percentage = 0
        for i in range(self.feature_nr):
            dim_number += 1
            percentage += self.evals[i]/sum
            if percentage >= 0.8:
                break
        self.reduced_dims = dim_number

    @staticmethod
    def dot_product(z, y):
        return z.dot(y)



