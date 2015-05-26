import sys

from cvm.svm import SVC

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

C = 100.0
gamma = 10.0
nmax = 2000


def parseData(line):
    fields = line.strip().split(",")
    label = int(fields[-1])
    feature = [float(feature) for feature in fields[:10]] #fields[0: -1]
    return LabeledPoint(1*(label==2), feature)


if __name__ == "__main__":
    if (len(sys.argv) != 1):
        print "Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g " + \
            "covertype.py"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("Cascade") \
        .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'

    trainRDD = sc.textFile('data/covertype/train.data', 24).map(parseData).cache()
    testRDD = sc.textFile('data/covertype/test.data', 24).map(parseData).cache()

    print trainRDD.first().label, trainRDD.first().features

    print "Number of train samples: ", trainRDD.count()
    print "Number of test samples: ", testRDD.count()

    print 'Fitting model'
    svm = SVC(gamma=gamma, C=C, nmax=nmax)
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
