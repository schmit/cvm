import sys

from cvm.svm import SVC
from cvm.kreg import KernelLogisticRegression

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

from sklearn.cross_validation import train_test_split

C = 100.0
gamma = 0.2
nmax = 20000


def parseData(line):
    fields = line.strip().split(",")
    label = int(fields[-1])
    feature = [float(feature)*5 for feature in fields[:10]] + [float(feature) for feature in fields[10:-1]] #fields[0: -1]
    return LabeledPoint(1*(label==2), feature)


if __name__ == "__main__":
    if (len(sys.argv) != 1):
        print "Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g " + \
            "covertype.py"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("Cascade") \
        .set("spark.executor.memory", "8g")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'

    trainRDD = sc.textFile('data/covertype/train.data').map(parseData).cache()
    
    testRDD = sc.textFile('data/covertype/test.data', 24).map(parseData).cache() 

    print trainRDD.first().label, trainRDD.first().features

    print "Number of train samples: ", trainRDD.count()
    print "Number of test samples: ", testRDD.count()

    print 'Fitting model'
    model = SVC(gamma=gamma, C=C, nmax=nmax)
    # model = KernelLogisticRegression(gamma=gamma, C=C, nmax=nmax)
    model.train(trainRDD)

    print 'Predicting outcomes training set'
    labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    print("Training Error = " + str(trainErr))

    print 'Predicting outcomes test set'
    labelsAndPredsTest = testRDD.map(lambda p: (p.label, model.predict(p.features)))
    testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print("Test Error = " + str(testErr))

    # clean up
    sc.stop()
