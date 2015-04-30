'''
Implementations of variations to SVM
'''

import numpy as np
import cvxpy as cvx

class SVM:
    '''
    Simple base class that implements Kernel-SVM-like prediction and decision
    functions
    '''
    def __init__(self, kernel, zero_tol=1.0e-3, **kwargs):
        self._kernel = kernel

        self.zero_tol = zero_tol

        for parameter, value in kwargs.items():
            setattr(self, parameter, value)

    def fit(self, X, y):
        raise NotImplementedError

    def decision_function(self, X):
        K = self._kernel(X, self.support_vectors_)
        return np.dot(K, self.coef_) + self.intercept_

    def predict(self, X):
        return np.sign(self.decision_function(X))


class L1SVM(SVM):
    '''
    Loss function:
        Hinge(w * Kernel) + reg * norm(w, 1)
    '''
    def fit(self, X, y):
        assert self.reg != None, 'Require <reg> regularization parameter set'

        # compute kernel matrix
        K = self._kernel(X, X)
        n = X.shape[0]

        # solve
        w = cvx.Variable(K.shape[1])
        b = cvx.Variable(1)

        obj = cvx.Minimize(
                    cvx.sum_entries(
                            cvx.max_elemwise(
                                    0,
                                    1 - cvx.mul_elemwise(y, K * w + b)
                            )
                    ) / n  + self.reg * cvx.norm(w, 1))
        prob = cvx.Problem(obj, [])
        prob.solve()

        # predict
        w = np.array(w.value)
        # clip w to 0
        w *= (np.abs(w) > self.zero_tol)

        self.support_ = np.flatnonzero(w)
        self.n_support_ = len(self.support_)
        self.coef_ = w[self.support_]
        self.intercept_ = b.value
        self.support_vectors_ = X[self.support_, :]


class ElasticSVM(SVM):
    '''
    Loss function:
        Hinge(w * Kernel) + alpha * reg * norm(w, 1) + (1-alpha) * reg * norm(w, 2)
    '''
    def fit(self, X, y, **kwargs):

        # set alpha if in kwargs
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = self.alpha

        assert self.reg != None, 'Require <reg> regularization parameter set'
        assert alpha != None, 'Require <alpha> parameter set'


        # compute kernel matrix
        K = self._kernel(X, X)
        n = X.shape[0]

        # solve
        w = cvx.Variable(K.shape[1])
        b = cvx.Variable(1)

        obj = cvx.Minimize(
                    cvx.sum_entries(
                            cvx.max_elemwise(
                                    0,
                                    1 - cvx.mul_elemwise(y, K * w + b)
                            )
                    ) / n
                    + alpha * self.reg * cvx.norm(w, 1)
                    + (1-alpha) * self.reg * cvx.norm(w, 2))

        prob = cvx.Problem(obj, [])
        prob.solve()

        # predict
        w = np.array(w.value)
        # clip w to 0
        w *= (np.abs(w) > self.zero_tol)

        self.support_ = np.flatnonzero(w)
        self.n_support_ = len(self.support_)
        self.coef_ = w[self.support_]
        self.intercept_ = b.value
        self.support_vectors_ = X[self.support_, :]
