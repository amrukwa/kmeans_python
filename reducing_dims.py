import numpy as np
import kmeans_functions as kmf
import scipy.linalg as linalg


class PCA:
    def __init__(self, x, reduce_to2="YES", reduced_dims=2):
        self.x = x
        self.reduced_dims = reduced_dims
        self.reduce_to2 = reduce_to2
        self.feature_nr = x.shape[1]
        self.datasize = x.shape[0]

    def pca(self):
        moved = kmf.subtract_mean(self.x)
        cov_moved = np.cov(np.transpose(moved))
        self.qr(cov_moved)
        importance = self.evals.argsort()[::-1]
        self.evals = self.evals[importance]
        self.evecs = self.evecs[importance]
        moved = self.dot_product(moved, self.evecs)
        if self.reduce_to2 == "NO":
            self._choose_dims()
        moved = moved[:, :self.reduced_dims]
        # for_visualization = pd.DataFrame(moved, columns=['x', 'y'])
        return moved  # reduced dim data

    def qr(self, matrix):
        a = matrix
        u = np.eye(a.shape[0])
        for i in range(100):
            prev = a
            q, r = np.linalg.qr(a)
            a = r.dot(q)
            u = matrix.dot(u)
            u = linalg.orth(u)
            if np.allclose(a, prev, 0.0001):
                break
        self.evecs = u
        self.evals = np.diag(a)

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
