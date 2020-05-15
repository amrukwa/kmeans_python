import numpy as np
from scipy.linalg import orth
import kmeans_functions as kmf


class PCA:
    def __init__(self, x, reduce_to2="NO", reduced_dims=2):
        self.x = x
        self.reduced_dims = reduced_dims
        self.reduce_to2 = reduce_to2
        self.feature_nr = x.shape[1]
        self.datasize = x.shape[0]

    def pca(self):
        moved = kmf.subtract_mean(self.x)
        cov_moved = np.cov(np.transpose(moved))
        self.evecs = np.eye(cov_moved.shape[0])
        for i in range(self.feature_nr):
            self._power_method(cov_moved, i)
        self.evals = np.array([self._rayleigh_quotient(cov_moved, i) for i in range(self.feature_nr)])
        importance = self.evals.argsort()[::-1]
        self.evals = self.evals[importance]
        self.evecs = self.evecs[importance]
        moved = self.dot_product(moved, self.evecs)
        if self.reduce_to2 == "NO":
            self._choose_dims()
        moved = moved[:, :self.reduced_dims]
        # for_visualization = pd.DataFrame(moved, columns=['x', 'y'])
        return moved  # reduced dim data

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
        sum_evals = np.sum(self.evals)
        dim_number = 0
        percentage = 0
        for i in range(self.feature_nr):
            dim_number += 1
            percentage += self.evals[i]/sum_evals
            if percentage >= 0.8:
                break
        self.reduced_dims = dim_number

    @staticmethod
    def dot_product(z, y):
        return z.dot(y)
