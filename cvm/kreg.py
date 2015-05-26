'''
Kernel regression methods using the Cascade
'''

import numpy as np
from sklearn.linear_model import LogisticRegression

from model import Model
from cascade import cascade
from kernel import rbf_kernel


class BaseKernelReg(Model):
    def __init__(self, kernel, gamma=1.0, nmax=1000):
        super(BaseKernelReg, self).__init__(nmax)

        if kernel == 'rbf':
            self.kernel = lambda X, Y: rbf_kernel(X, Y, gamma)
        else:
            raise NotImplementedError('Kernel not supported')


    def train(self, labeledPoints):
        labeledPoints = cascade(labeledPoints, self._reduce, self.nmax)
        self.X, y = self._readiterator(labeledPoints)
        self.model, self.support = self._fit(self.X, y)

    def predict(self, features):
        f = features.reshape(1, len(features))
        k = self.kernel(f, self.X)
        return self.model.predict(k)

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)

        # if few datapoints, don't thin further
        if len(y) < self.nmax/2:
            return self._returniterator(xrange(len(y)), X, y)

        _, support = self._fit(X, y)
        if len(support) < len(y) / 2:
            return self._returniterator(support, X, y)

        vectors_lost = len(support) - len(y)/2
        self.lost += vectors_lost
        print 'Warning: {} relevant vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(support, len(y) / 2, replace=False)
        return self._returniterator(random_indices, X, y)

    def _fit(self, X, y):
        K = self.kernel(X, X)
        model = self.create_model()
        model.fit(K, y)
        support = np.nonzero(np.sum(np.abs(model.coef_), 0))[0]
        return model, support


class KernelLogisticRegression(BaseKernelReg):
    def __init__(self, kernel='rbf', gamma=1.0, C=1.0, nmax=1000):
        super(KernelLogisticRegression, self).__init__(kernel, gamma, nmax)
        self.create_model = lambda : LogisticRegression('l1', C=C)


