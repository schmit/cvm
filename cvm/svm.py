'''
Implementations of variations to SVM
'''

import numpy as np
import cvxpy as cvx

class L1SVM:
    def __init__(self, kernel, l1_reg=1.0, zero_tol=1.0e-3):
        self._kernel = kernel
        self.l1_reg = l1_reg

        self.zero_tol = zero_tol

    def fit(self, X, y):
        # compute kernel matrix
        K = self._kernel(X, X)

        # solve
        w = cvx.Variable(K.shape[1])
        b = cvx.Variable(1)

        obj = cvx.Minimize(
                    cvx.sum_entries(
                            cvx.max_elemwise(
                                    0,
                                    1 - cvx.mul_elemwise(y, K * w + b)
                            )
                    ) + self.l1_reg * cvx.norm(w, 1))
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

    def decision_function(self, X):
        K = self._kernel(X, self.support_vectors_)
        return np.dot(K, self.coef_) + self.intercept_

    def predict(self, X):
        return np.sign(self.decision_function(X))

