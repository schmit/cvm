'''
WIP Combines Cascade and Models
'''

import random
import numpy as np
from sklearn import svm
from pyspark.mllib.regression import LabeledPoint

from cascade import cascade

class BaseSVM(object):
    def __init__(self, nmax):
        self.nmax = nmax
        # self.create_model = lambda : svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        self.lost_svs = 0

    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        X, y = self._readiterator(labeledPoints)
        self.model = self.create_model()
        self.model.fit(X, y)

    def simple_train(self, labeledPoints):
        X, y = self._readiterator(labeledPoints.collect())
        self.model = self.create_model()
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)
        model = self.create_model()
        model.fit(X, y)
        if len(model.support_) < self.nmax/2: #len(y) / 2:
            return self._returniterator(model.support_, X, y)

        vectors_lost = len(model.support_) - len(y)/2
        self.lost_svs += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, self.nmax / 2, replace=False)
        return self._returniterator(random_indices, X, y) #len(y) / 2

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


class SVC(BaseSVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0, nmax=2000):
        super(SVC, self).__init__(nmax)
        self.create_model = lambda : svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)


class NuSVC(BaseSVM):
    def __init__(self, nu=0.3, kernel='rbf', degree=3, gamma=1.0, nmax=2000):
        super(NuSVC, self).__init__(nmax)
        self.create_model = lambda : svm.NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma)


class SVR(BaseSVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0, nmax=2000):
        super(SVR, self).__init__(nmax)
        self.create_model = lambda : svm.SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)


class NuSVR(BaseSVM):
    def __init__(self, nu=0.3, kernel='rbf', degree=3, gamma=1.0, nmax=2000):
        super(SVR, self).__init__(nmax)
        self.create_model = lambda : svm.NuSVR(nu=nu, kernel=kernel, degree=degree, gamma=gamma)


class RandomSVM(BaseSVM):
    def __init__(self, kernel='rbf', degree=3, C=1.0, gamma=1.0, nmax=2000):
        self.nmax = nmax
        self.create_model = lambda : svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

        self.lost_svs = 0

    def _reduce(self, iterator):
        for elem in iterator:
            if random.random() < 0.5:
                yield elem

