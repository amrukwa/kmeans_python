import sklearn.metrics as metrics
import numpy as np


def calculate_score_1(data, labels_true):
    labels_pred = data
    score = metrics.adjusted_rand_score(labels_true, labels_pred)
    return score


def split_into_sets(data, k):
    pass


def validate(estimator, data, k=10):
    np.random.shuffle(data)
    split_into_sets(data, k)
    score = 0
    labels_true = 0
    for i in range(k):
        score += calculate_score_1(data, labels_true)
    return score/k
