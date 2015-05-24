import sys
from sklearn import svm as sklearnSVM
from cvm.svm import NuSVC, SVC
from sklearn.cross_validation import train_test_split

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint

def parseData(line):
    fields = line.strip().split(",")
    label = int(int(fields[-1]) == 2)
    feature = [float(feature) for feature in fields[: -1]] #fields[0: -1]
    return LabeledPoint(label, feature)


if __name__ == "__main__":
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
    data = sc.textFile('../data/covertype/shuffle_scaled_covtype.data').collect()
   
    # Split data aproximately into training (60%) and test (40%)
    data, throwaway = train_test_split(data, train_size = 0.1)  
    train, test = train_test_split(data, train_size = 0.6)    

    trainRDD = sc.parallelize(train).map(parseData).cache() # filter(filter2classes)
    testRDD = sc.parallelize(test).map(parseData).cache() # filter(filter2classes).

    print "Size of train set: ", trainRDD.count()
    print "Size of test set: ", testRDD.count()

    print 'Fitting model'
    svm = SVC(C=100.0, gamma=10.0, nmax=7000)
    svm.loopy_train(trainRDD)

    print 'Predicting outcomes training set'
    labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, svm.predict(p.features)))
    trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    print("Training Error = " + str(trainErr))

    print 'Predicting outcomes test set'
    labelsAndPredsTest = testRDD.map(lambda p: (p.label, svm.predict(p.features)))
    testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print("Test Error = " + str(testErr))


    # # # ############################################################################################
    # Subsample and split data aproximately into training (60%) and test (40%)
    small_trainRDD = sc.parallelize(train).map(parseData).cache()    #sc.textFile('../data/covertype/trainCovtype.data').map(parseData).cache()
    small_testRDD = sc.parallelize(test).map(parseData).cache()     #sc.textFile('../data/covertype/testCovtype.data').map(parseData).cache()

    print "Size of downsampled train set: ", small_trainRDD.count()
    print "Size of downsapled test set: ", small_testRDD.count()

    print 'Fitting a simple model on heavily downsapled data'
    simple_svm = SVC(C=100.0, gamma=10.0)
    simple_svm.simple_train(small_trainRDD)

    print 'Predicting outcomes small training set'
    simple_labelsAndPredsTrain = small_trainRDD.map(lambda p: (p.label, simple_svm.predict(p.features)))
    simple_trainErr = simple_labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(small_trainRDD.count())
    print("Training Error = " + str(simple_trainErr))

    print 'Predicting outcomes small test set'
    simple_labelsAndPredsTest = small_testRDD.map(lambda p: (p.label, simple_svm.predict(p.features)))
    simple_testErr = simple_labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(small_testRDD.count())
    print("Test Error = " + str(simple_testErr))

    # clean up
    sc.stop()

