'''
Implementation of recursive cascading for SVM models

Based on:
Graf, Hans P., et al. "Parallel support vector machines: The cascade svm."
Advances in neural information processing systems. 2004.
'''

from __future__ import division

import numpy as np

def cascade(labeledPointRDD, reducer, nmax):
    n = labeledPointRDD.count()
    numPartitions = int(2**(np.ceil(np.log(n / nmax)/np.log(2.0))))
    # numLevels = int(np.round(np.log(numPartitions) / np.log(2) + 1))
    leafsRDD = labeledPointRDD.repartition(numPartitions)

    while numPartitions > 1:
        print 'Currently {} partitions left'.format(numPartitions)
        print 'Size of data: {}'.format(leafsRDD.count())

        numPartitions = int(numPartitions / 2)

        # need cache else lazy evaluation is killing
        leafsRDD = leafsRDD.mapPartitions(reducer, True) \
                           .coalesce(numPartitions) \
                           .cache()

    return leafsRDD.collect()


