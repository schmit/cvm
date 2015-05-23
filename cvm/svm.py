'''
WIP Combines Cascade and Models
'''

import random
import numpy as np
from sklearn import svm
from pyspark.mllib.regression import LabeledPoint

from cascade import *
    

# def readiterator(iterator):
#     ys = []
#     xs = []
#     for elem in iterator:
#         ys.append(elem.label)
#         xs.append(elem.features)

#     X = np.array(xs)
#     y = np.array(ys)
#     return X, y

# def returniterator(indices, X, y):
#     for i in indices:
#         yield LabeledPoint(y[i], X[i])


class BaseSVM(object):
    def __init__(self, nmax):
        self.nmax = nmax
        # self.create_model = lambda : svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        self.lost_svs = 0

    def train(self, labeledPoints):
        labeledPoints = self._repartition(labeledPoints, self.nmax)
        labeledPoints = cascade(labeledPoints, self._reduce)
        X, y = self._readiterator(labeledPoints)
        self.model = self.create_model()
        self.model.fit(X, y)

    def simple_train(self, labeledPoints):
        X, y = self._readiterator(labeledPoints.collect())
        self.model = self.create_model()
        self.model.fit(X, y)

    def loopy_train(self, labeledPoints):
        #stopCondition=False
        labeledPoints0 = self._repartition(labeledPoints, self.nmax).cache()
        labeledPoints = labeledPoints0
        for i in xrange(3): # while stopCondition==False:
            filteredLabeledPoints = cascade(labeledPoints, self._reduce)
            X, y = self._readiterator(filteredLabeledPoints)
            Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
            _, unique_idx = np.unique(Z, return_index=True)
            X = X[np.sort(unique_idx)]
            y = y[np.sort(unique_idx)]
            model = self.create_model()
            model.fit(X, y)

            labelsAndPredsTrain = labeledPoints0.map(lambda p: (p.label, model.predict(p.features)))
            trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(labeledPoints0.count())
            print("[IMPORTANT] Training Error at round (" + str(i) + ") = " + str(trainErr))

            sharedSVIter = self._returniterator(model.support_, X, y)
            self.shared_SVs, self.shared_SVLab = self._readiterator(sharedSVIter)

            print ("Number of shared SVs after round (" + str(i) + ") = " + str( self.shared_SVs.shape[0]))
            labeledPoints = combineSharedSVs(labeledPoints, self._reduce_with_shared_SVs)
            self.lost_svs = 0
            fractionOfNewSVs = fraction_new_svs(labeledPoints, self._fracNewSVs)
            print "Fractions of partition specific SVs not in the shared SV pool: "
            print fractionOfNewSVs
            #stopCondition = np.all(fractionOfNewSVs < 0.1)  

        filteredLabeledPoints = cascade(labeledPoints, self._reduce)
        X, y = self._readiterator(filteredLabeledPoints)
        Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
        _, unique_idx = np.unique(Z, return_index=True)
        X = X[np.sort(unique_idx)]
        y = y[np.sort(unique_idx)]
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
        Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
        _, unique_idx = np.unique(Z, return_index=True)
        X = X[np.sort(unique_idx)]
        y = y[np.sort(unique_idx)]

        model = self.create_model()
        model.fit(X, y)
        if len(model.support_) < self.nmax/2: #len(y) / 2:
            return self._returniterator(model.support_, X, y)

        vectors_lost = len(model.support_) - self.nmax/2
        self.lost_svs += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, self.nmax / 2, replace=False)
        return self._returniterator(random_indices, X, y) #len(y) / 2

    def _reduce_with_shared_SVs(self, iterator):
        X, y = self._readiterator(iterator)

        X = np.vstack([X, self.shared_SVs])
        y = np.asarray(y.tolist() + self.shared_SVLab.tolist())
        Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
        _, unique_idx = np.unique(Z, return_index=True)

        X = X[np.sort(unique_idx)]
        y = y[np.sort(unique_idx)]

        model = self.create_model()
        model.fit(X, y)
        newLabeledSVIter = self._returniterator(model.support_, X, y)
        
        if len(model.support_) < self.nmax/2: 
            return newLabeledSVIter

        vectors_lost = len(model.support_) - self.nmax/2
        self.lost_svs += vectors_lost
        print 'Warning: {} relevant support vectors thrown away!'.format(vectors_lost)
        random_indices = np.random.choice(model.support_, self.nmax / 2, replace=False)

        return self._returniterator(random_indices, X, y) 

    def _fracNewSVs(self, iterator):
        X, y = self._readiterator(iterator)

        X = np.vstack([X, self.shared_SVs])
        y = np.asarray(y.tolist() + self.shared_SVLab.tolist())
        Z = X.ravel().view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
        _, unique_idx = np.unique(Z, return_index=True)

        X = X[np.sort(unique_idx)]
        y = y[np.sort(unique_idx)]

        model = self.create_model()
        model.fit(X, y)
        newLabeledSVIter = self._returniterator(model.support_, X, y)

        newSVs, newSVlabels = self._readiterator(newLabeledSVIter)
        dt = np.dtype((np.void, newSVs.dtype.itemsize * newSVs.shape[1]))
        sv_intersection_idx = np.nonzero(np.in1d(newSVs.view(dt).reshape(-1), self.shared_SVs.view(dt).reshape(-1)))[0]
        numIntersectSVs = sv_intersection_idx.shape[0]
        print 'Size of union of partition and shared SVs: ', X.shape[0]
        print "# SVs trained on a union of partition and shared SVs: ", newSVs.shape[0]
        print "# of intersecting SVs: ", numIntersectSVs
        frac_new_svs = (newSVs.shape[0] - numIntersectSVs)*1.0/self.shared_SVs.shape[0]
        print "Ration # new SVs / # shared SVs: ", frac_new_svs
        return iter([frac_new_svs])


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

