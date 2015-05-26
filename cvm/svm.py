'''
Implements Support vector methods using the Cascade
'''

import random
import numpy as np
from sklearn import svm

from cascade import *
from model import Model

class BaseSVM(Model):
    def train(self, labeledPoints):
        labeledPoints = self._repartition(labeledPoints, self.nmax)
        labeledPoints = cascade(labeledPoints, self._reduce)
        X, y = self._readiterator(labeledPoints)
        self.model = self.create_model()
        self.model.fit(X, y)
        self.sharedSVs = None
        self.sharedSVLab = None

    def simple_train(self, labeledPoints):
        X, y = self._readiterator(labeledPoints.collect())
        self.model = self.create_model()
        self.model.fit(X, y)
        print '[Simple Model] number of support vectors = ', len(self.model.support_)


    def loopy_train(self, labeledPoints, numLoops=5):
        labeledPoints0 = self._repartition(labeledPoints, self.nmax).cache()
        labeledPoints = labeledPoints0
        for i in xrange(numLoops): 
            filteredLabeledPoints = cascade(labeledPoints, self._reduce)
            X, y = self._readiterator(filteredLabeledPoints)
            X, y = self._get_unique_rows(X, y)
            model = self.create_model()
            model.fit(X, y)
            labelsAndPredsTrain = labeledPoints0.map(lambda p: (p.label, model.predict(p.features)))
            trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(labeledPoints0.count())
            print("[IMPORTANT] Training Error at round (" + str(i) + ") = " + str(trainErr))

            sharedSVIter = self._returniterator(model.support_, X, y)
            self.sharedSVs, self.sharedSVLab = self._readiterator(sharedSVIter)
            print ("Number of shared SVs after round (" + str(i) + ") = " + str( self.sharedSVs.shape[0]))
            
            labeledPoints = combine_shared_svs(labeledPoints0, self._combine_with_shared_svs)
            self.lost_svs = 0

            # The following 3 lines can be commented out if you don't want to print out progress 
            fractionOfNewSVs = fraction_new_svs(labeledPoints0, self._frac_new_svs)
            print "Fractions of partition specific SVs not in the shared SV pool: "
            print fractionOfNewSVs

        filteredLabeledPoints = cascade(labeledPoints, self._reduce)
        X, y = self._readiterator(filteredLabeledPoints)
        X, y = self._get_unique_rows(X, y)
        self.model = self.create_model()
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict(features)

    def _repartition(self, labeledPointRDD, nmax):
        n = labeledPointRDD.count()
        numPartitions = int(2**(np.ceil(np.log(n / nmax)/np.log(2.0))))
        leafsRDD = labeledPointRDD.repartition(numPartitions)
        return leafsRDD

    def _reduce(self, iterator):
        X, y = self._readiterator(iterator)
        X, y = self._get_unique_rows(X, y)

        # if few datapoints, don't thin further
        if len(y) < self.nmax/2:
            return self._returniterator(xrange(len(y)), X, y)

        model = self.create_model()
        model.fit(X, y)
        if len(model.support_) < self.nmax/2: #len(y) / 2:
            return self._returniterator(model.support_, X, y)

        vectors_lost = len(model.support_) - self.nmax/2
        self.lost += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, self.nmax / 2, replace=False)
        return self._returniterator(random_indices, X, y) 

    def _combine_with_shared_svs(self, iterator):
        X, y = self._readiterator(iterator)
        X = np.vstack([X, self.sharedSVs])
        y = np.asarray(y.tolist() + self.sharedSVLab.tolist())
        X, y = self._get_unique_rows(X, y)
        return self._returniterator(range(X.shape[0]), X, y) # self._returniterator(random_indices, X, y) 

    def _frac_new_svs(self, iterator):
        X, y = self._readiterator(iterator)
        X = np.vstack([X, self.sharedSVs])
        y = np.asarray(y.tolist() + self.sharedSVLab.tolist())
        X, y = self._get_unique_rows(X, y)

        model = self.create_model()
        model.fit(X, y)
        newLabeledSVIter = self._returniterator(model.support_, X, y)

        newSVs, newSVlabels = self._readiterator(newLabeledSVIter)
        dt = np.dtype((np.void, newSVs.dtype.itemsize * newSVs.shape[1]))
        sv_intersection_idx = np.nonzero(np.in1d(newSVs.view(dt).reshape(-1), self.sharedSVs.view(dt).reshape(-1)))[0]
        numIntersectSVs = sv_intersection_idx.shape[0]

        print 'Size of union of partition and shared SVs: ', X.shape[0]
        print "# SVs trained on a union of partition and shared SVs: ", newSVs.shape[0]
        print "# of intersecting SVs: ", numIntersectSVs
        frac_new_svs = (newSVs.shape[0] - numIntersectSVs)*1.0/self.sharedSVs.shape[0]
        print "Ration # new SVs / # shared SVs: ", frac_new_svs
        return iter([frac_new_svs])

    def _get_unique_rows(self, X, y):
        Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
        _, unique_idx = np.unique(Z, return_index=True)
        X = X[np.sort(unique_idx)]
        y = y[np.sort(unique_idx)]
        return X, y


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

    def _reduce(self, iterator):
        for elem in iterator:
            if random.random() < 0.5:
                yield elem

