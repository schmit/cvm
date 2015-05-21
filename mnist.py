#!/usr/bin/env python

'''
Run using:
[SparkDir]/spark/bin/spark-submit --driver-memory 2g mnist.py
'''

import sys

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

from cvm.svm import KernelSVM

def objective(x):
    # prediction objective
    return 1*(x>4)

def parseData(line, obj):
    fields = line.strip().split(',')
    return LabeledPoint(obj(int(fields[0])), [int(v) for v in fields[1:]])


if __name__ == "__main__":
    if (len(sys.argv) != 1):
        print "Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g " + \
            "mnist.py"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("Cascade") \
        .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'
    trainRDD = sc.textFile('data/mnist/mnist_train.csv').map(lambda line: parseData(line, objective)).cache()
    testRDD = sc.textFile('data/mnist/mnist_test.csv').map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'
    svm = KernelSVM(gamma=0.0000000001, C=1000000.0)
    svm.train(trainRDD)

    print 'Predicting outcomes training set'
    labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, svm.predict(p.features)))
    trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    print("Training Error = " + str(trainErr))

    print 'Predicting outcomes test set'
    labelsAndPredsTest = testRDD.map(lambda p: (p.label, svm.predict(p.features)))
    testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print("Test Error = " + str(testErr))

    # clean up
    sc.stop()