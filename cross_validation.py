import sklearn.metrics as metrics
import numpy as np


class KFold:
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def validate(self, data, labels_true, estimator):
        indices = [i for i in range(data.shape[0])]
        np.random.shuffle(indices)
        split = np.array_split(indices, self.n_splits)

        for array in split:
            test = np.array([data[i] for i in range(array.shape[0])])
            train = np.array([data[i] for i in range(array.shape[0], data.shape[0])])

        score = 0
        return score / self.n_splits


def calculate_score(data, labels_true):
    labels_pred = data
    score = metrics.adjusted_rand_score(labels_true, labels_pred)
    return score
