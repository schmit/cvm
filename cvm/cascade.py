'''
Implementation of recursive cascading for SVM models

Based on:
Graf, Hans P., et al. "Parallel support vector machines: The cascade svm."
Advances in neural information processing systems. 2004.
'''

from __future__ import division

import numpy as np


class Cascade:
    '''
    Approximates SVM by splitting data in small chunks, running SVM locally,
    and combining support vectors.

    This could be done in parallel, though this is a serial implementation
    '''

    def __init__(self, model, nmax):
        self.model = model
        self.nmax = nmax

    def train(self, labeledPointRDD):
        n = labeledPointRDD.count()
        numPartitions = int(2**(np.ceil(np.log(n / self.nmax)/np.log(2.0))))
        # numLevels = int(np.round(np.log(numPartitions) / np.log(2) + 1))
        leafsRDD = labeledPointRDD.repartition(numPartitions)

        while numPartitions > 1:
            numPartitions = int(numPartitions / 2)
            leafsRDD = leafsRDD.mapPartitions(self.model.reduce)
            leafsRDD = leafsRDD.coalesce(numPartitions)

        self.model.train(leafsRDD.collect())


    def predict(self, features):
        return self.model.predict(features)

