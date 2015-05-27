import sys
import time
from sklearn import svm as sklearnSVM
from cvm.svm import NuSVC, SVC
from sklearn.cross_validation import train_test_split
from cvm.cascade import readiterator

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

def parseData(line):
    fields = line.strip().split(",")
    label = int(int(fields[-1]) == 2)
    feature = [float(feature) for feature in fields[: -1]] 
    return LabeledPoint(label, feature)


if __name__ == "__main__":
    C1 = 100.0
    gamma1 = 10.0
    nmax1 = 2000

    if (len(sys.argv) != 1):
        print "Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g " + \
            "./bin/covertype.py"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("Cascade") \
        .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'
    data = sc.textFile('data/covertype/shuffle_scaled_covtype.data').collect()
   
    # Split data aproximately into training (60%) and test (40%)
    keep, throwaway = train_test_split(data, train_size = 0.2, random_state=0)  
    train, test = train_test_split(keep, train_size = 0.6, random_state=0)    

    trainRDD = sc.parallelize(train).map(parseData).cache() # filter(filter2classes)
    testRDD = sc.parallelize(test).map(parseData).cache() # filter(filter2classes).

    print "Size of train set: ", trainRDD.count()
    print "Size of test set: ", testRDD.count()

    print 'Fitting cascade model'
    print("Model parameters: C=" + str(C1) + "; gamma=" + str(gamma1) + "; nmax=" + str(nmax1))
    svm = SVC(C=C1, gamma=gamma1, nmax=nmax1)
    start = time.time()
    svm.train(trainRDD)
    #svm.loopy_train(trainRDD)
    end = time.time()
    print("Cascade SVM train on " + str(trainRDD.count()) + " samples took " + str(end - start) + " time.")


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

