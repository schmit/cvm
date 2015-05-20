'''
Implementation of recursive cascading for SVM models

Based on:
Graf, Hans P., et al. "Parallel support vector machines: The cascade svm."
Advances in neural information processing systems. 2004.
'''

import numpy as np

class CascadeSVM:
    '''
    Approximates SVM by splitting data in small chunks, running SVM locally,
    and combining support vectors.

    This could be done in parallel, though this is a serial implementation
    '''
    def __init__(self, model, nx=500, n_iter=1, nmax=1000):
        self.nx = nx
        self.basemodel = model
        self._split = 0
        self._max_splits = 50
        self.n_iter = 1     # TO FIX: currently using more iterations is broken
        self.nmax = nmax

    def fit(self, X, y):
        '''
        Fit cascading SVM model
        '''

        self._max_level = np.ceil((np.log(len(X)) - np.log(self.nx))/np.log(2))

        # initialize support vectors to be empty
        self.support_vectors_ = X[[], :]
        self.support_outcomes_ = y[[]]

        for iter in xrange(self.n_iter):
            self.support_vectors_, self.support_outcomes_, model = self._recurse(X, y, 0)

        self.model = model

        # set some parameters such that we can use it for plotting etc.
        self.support_ = model.support_
        self.decision_function = model.decision_function
        self.n_support_ = model.n_support_

    def predict(self, Xnew):
        '''
        Predict using new data
        '''
        return self.model.predict(Xnew)

    def _recurse(self, X, y, level):
        n = len(X)
        self._split += 1

        # if number of samples is small enough, fit SVM
        if n < self.nx:
            # TO FIX: duplicates support vectors
            # add current support vectors to data
            # X = np.vstack((X, self.support_vectors_))
            # y = np.hstack((y, self.support_outcomes_))
            return self._fit(X, y, level)

        # otherwise split data in 2
        print 'split: {} \t n = {}'.format(self._split, n)
        leftX, lefty, _ = self._recurse(X[:n/2, :], y[:n/2], level+1)
        rightX, righty, _ = self._recurse(X[n/2:, :], y[n/2:], level+1)

        # recurse and combine using another SVM fit
        combinedX = np.vstack((leftX, rightX))
        combinedy = np.hstack((lefty, righty))
        # cut observations if too many
        if len(combinedy) > self.nmax:
            combinedy = combinedy[:self.nmax]
            combinedX = combinedX[:self.nmax]

        return self._fit(combinedX, combinedy, level)

    def _fit(self, X, y, level):
        # print to show what's going on
        print 'fitting svm with {} observations at level {}'.format(len(X), level)

        # fit a simple svm
        model = self.basemodel()

        model.fit(X, y)

        # return the supports and the model
        return X[model.support_, :], y[model.support_], model


class DynamicCascadeSVM(CascadeSVM):
    '''
    CascadeSVM that changes parameter alpha based on depth of recursion:
    useful for ElasticSVM so that at the leafs we can use predominantly L1
    penalty, while at the final step, we mostly use L2 regularization.
    '''
    def _fit(self, X, y, level):
        # print to show what's going on
        print 'fitting svm with {} observations at level {}'.format(len(X), level)

        # fit a simple svm
        model = self.basemodel()

        model.fit(X, y, alpha= (level+1.0) / (self._max_level+2.0))

        # return the supports and the model
        return X[model.support_, :], y[model.support_], model


