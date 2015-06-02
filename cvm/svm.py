'''
Implements Support vector methods using the Cascade
'''

import random
import numpy as np
from sklearn import svm

from model import Model
from cascade import cascade

class BaseSVM(Model):
    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        X, y = self._readiterator(labeledPoints)
        self.model = self.create_model()
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)

        # if few datapoints, don't thin further
        if len(y) < self.nmax/2:
            return self._returniterator(xrange(len(y)), X, y)

        model = self.create_model()
        model.fit(X, y)
        if len(model.support_) < self.max_sv:
            return self._returniterator(model.support_, X, y)

        vectors_lost = len(model.support_) - self.max_sv

        self.lost += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, self.max_sv, replace=False)
        return self._returniterator(random_indices, X, y)


class SVC(BaseSVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0,
                 nmax=2000, max_sv=0.5):
        super(SVC, self).__init__(nmax, max_sv)
        self.create_model = lambda : svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)


class NuSVC(BaseSVM):
    def __init__(self, nu=0.3, kernel='rbf', degree=3, gamma=1.0,
                 nmax=2000, max_sv=0.5):
        super(NuSVC, self).__init__(nmax, max_sv)
        self.create_model = lambda : svm.NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma)


class SVR(BaseSVM):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0,
                 nmax=2000, max_sv=0.5):
        super(SVR, self).__init__(nmax, max_sv)
        self.create_model = lambda : svm.SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)


class NuSVR(BaseSVM):
    def __init__(self, nu=0.3, kernel='rbf', degree=3, gamma=1.0,
                 nmax=2000, max_sv=0.5):
        super(SVR, self).__init__(nmax, max_sv)
        self.create_model = lambda : svm.NuSVR(nu=nu, kernel=kernel, degree=degree, gamma=gamma)

