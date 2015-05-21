'''
WIP Combines Cascade and Models
'''

import random
import numpy as np
from sklearn import svm
from pyspark.mllib.regression import LabeledPoint

from cascade import cascade
from reducers import randomreduce

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, nmax=800, gamma=1.0, degree=3):
        self.nmax = nmax
        self.create_model = lambda : svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

        self.lost_svs = 0

    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        X, y = self._readiterator(labeledPoints)
        self.model = self.create_model()
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)
        model = self.create_model()
        model.fit(X, y)
        if len(model.support_) < len(y) / 2:
            return self._returniterator(model.support_, X, y)

        vectors_lost = len(model.support_) - len(y)/2
        self.lost_svs += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, len(y) / 2, replace=False)
        return self._returniterator(random_indices, X, y)

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


class RandomSVM:
    def __init__(self, kernel='rbf', C=1.0, nmax=800, gamma=1.0, degree=3):
        self.nmax = nmax
        self.create_model = lambda : svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

        self.lost_svs = 0

    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        X, y = self._readiterator(labeledPoints)

        self.model = self.create_model()
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _reduce(self, iterator):
        for elem in iterator:
            if random.random() < 0.5:
                yield elem

    def _readiterator(self, iterator):
        ys = []
        xs = []
        for elem in iterator:
            ys.append(elem.label)
            xs.append(elem.features)

        X = np.array(xs)
        y = np.array(ys)
        return X, y
