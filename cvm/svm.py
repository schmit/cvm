'''
WIP Combines Cascade and Models
'''

import numpy as np
from sklearn import svm
from pyspark.mllib.regression import LabeledPoint


from cascade import cascade

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, nmax=500, gamma=1.0, degree=3):
        self.nmax = nmax
        self.model = svm.SVC(kernel='rbf', C=C, gamma=gamma)

        self.lost_svs = 0

    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        X, y = self._readiterator(labeledPoints)
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)
        self.model.fit(X, y)
        if len(self.model.support_) < len(y) / 2:
            return self._returniterator(self.model.support_, X, y)

        vectors_lost = len(self.model.support_) - len(y)/2
        self.lost_svs += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(self.model.support_, len(y) / 2, replace=False)
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
