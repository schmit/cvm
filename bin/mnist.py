#!/usr/bin/env python

'''
Run using:
[SparkDir]/spark/bin/spark-submit --driver-memory 2g mnist.py
'''

import sys
import time

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

from cvm.svm import SVC
from cvm.kreg import KernelLogisticRegression

# Set parameters here:
NMAX = 10000
GAMMA = 0.02
C = 1.0

def objective(x):
    # prediction objective
    return x

def parseData(line, obj):
    fields = line.strip().split(',')
    return LabeledPoint(obj(int(fields[0])), [float(v)/255.0 for v in fields[1:]])


if __name__ == "__main__":
    if (len(sys.argv) != 1):
        print "Usage: [SPARKDIR]/bin/spark-submit --driver-memory 2g " + \
            "mnist.py"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("Cascade") \
        .set("spark.executor.memory", "2g") \
        .set("spark.kryoserializer.buffer.mb", "128")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'
    time_start = time.time()
    trainRDD = sc.textFile('data/mnist/train.csv').map(lambda line: parseData(line, objective)).cache()
    testRDD = sc.textFile('data/mnist/test.csv').map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'
    model = SVC(gamma=GAMMA, C=C, nmax=NMAX)
    # model = KernelLogisticRegression(gamma=0.01, C=2.0, nmax=3000)

    model.train(trainRDD)
    print("Time: {:2.2f}".format(time.time() - time_start))

    print 'Predicting outcomes training set'
    labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    print("Training Error = " + str(trainErr))
    print("Time: {:2.2f}".format(time.time() - time_start))

    print 'Predicting outcomes test set'
    labelsAndPredsTest = testRDD.map(lambda p: (p.label, model.predict(p.features)))
    testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print("Test Error = " + str(testErr))
    print("Time: {:2.2f}".format(time.time() - time_start))

    # clean up
    sc.stop()
