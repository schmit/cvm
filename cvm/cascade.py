'''
Implementation of recursive cascading for SVM models
'''

import numpy as np

class CascadeSVM:
    def __init__(self, model, nx=500):
        self.nx = nx
        self.basemodel = model
        self._split = 0
        self._max_splits = 50

    def fit(self, X, y):
        X_support, y_support, model = self._recurse(X, y)
        self.model = model

        # set some parameters such that we can use it for plotting etc.
        self.support_ = model.support_
        self.support_vectors_ = model.support_vectors_
        self.decision_function = model.decision_function
        self.n_support_ = model.n_support_

    def predict(self, Xnew):
        return self.model.predict(Xnew)

    def _recurse(self, X, y):
        n = len(X)
        self._split += 1

        # if number of samples is small enough, fit SVM
        if (n < self.nx) or (self._split > self._max_splits):
            return self._fit(X, y)

        # otherwise split data in 2
        print 'split: {} \t n = {}'.format(self._split, n)
        leftX, lefty, _ = self._recurse(X[:n/2, :], y[:n/2])
        rightX, righty, _ = self._recurse(X[n/2:, :], y[n/2:])

        # recurse and combine using another SVM fit
        return self._fit(np.vstack((leftX, rightX)), np.hstack((lefty, righty)))

    def _fit(self, X, y):
        # print to show what's going on
        print 'fitting svm with {} observations'.format(len(X))

        # fit a simple svm
        model = self.basemodel()
        model.fit(X, y)

        # return the supports and the model
        return X[model.support_, :], y[model.support_], model
