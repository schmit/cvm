'''
Implementation of recursive cascading for SVM models

Based on:
Graf, Hans P., et al. "Parallel support vector machines: The cascade svm."
Advances in neural information processing systems. 2004.
'''

from __future__ import division

import numpy as np

def cascade(leafsRDD, reducer):
    numPartitions = leafsRDD.getNumPartitions()
    while numPartitions > 1:
        print 'Currently {} partitions left'.format(numPartitions)
        print 'Size of data: {}'.format(leafsRDD.count())

        numPartitions = int(numPartitions / 2)

        # need cache else lazy evaluation is killing
        leafsRDD = leafsRDD.mapPartitions(reducer, True) \
                           .coalesce(numPartitions) \
                           .cache()
    return leafsRDD.collect()

def combine_shared_svs(labeledPointsRDD, reducer):
    newLabeledPointsRDD = labeledPointsRDD.mapPartitions(reducer, True).cache()
    return newLabeledPointsRDD

def fraction_new_svs(labeledPointsRDD, reducer):
    frac = labeledPointsRDD.mapPartitions(reducer, True).cache()
    return frac.collect()


def readiterator(iterator):
    ys = []
    xs = []
    for elem in iterator:
        ys.append(elem.label)
        xs.append(elem.features)

    X = np.array(xs)
    y = np.array(ys)
    return X, y
