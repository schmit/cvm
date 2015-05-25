''' Abstract Model class '''

import numpy as np

from pyspark.mllib.regression import LabeledPoint


class Model(object):
    def __init__(self, nmax):
        self.nmax = nmax
        self.lost = 0

    def train(self, labeledPoints):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def _reduce(self, iterator):
        raise NotImplementedError

    def _readiterator(self, iterator):
        ys = []
        xs = []
        for elem in iterator:
            ys.append(elem.label)
            xs.append(elem.features)

        X = np.array(xs)
        y = np.array(ys)
        return X, y

    def _returniterator(self, indices, X, y):
        for i in indices:
            yield LabeledPoint(y[i], X[i])