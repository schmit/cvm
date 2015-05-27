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
    nmax1 = 20000

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
   
    Split data aproximately into training (60%) and test (40%)
    keep, throwaway = train_test_split(data, train_size = 0.2, random_state=0)  
    train, test = train_test_split(keep, train_size = 0.6, random_state=0)    

    small_trainRDD = sc.parallelize(train).map(parseData).cache()    #sc.textFile('../data/covertype/trainCovtype.data').map(parseData).cache()
    small_testRDD = sc.parallelize(test).map(parseData).cache()     #sc.textFile('../data/covertype/testCovtype.data').map(parseData).cache()

    print "Size of downsampled train set: ", small_trainRDD.count()
    print "Size of downsapled test set: ", small_testRDD.count()

    print 'Fitting a simple model on heavily downsapled data'
    print("Model parameters: C=" + str(C1) + "; gamma=" + str(gamma1))
    #simple_svm = SVC(C=C1, gamma=gamma1)
    X, y = readiterator(small_trainRDD.collect())
    simple_svm = sklearnSVM.SVC(C=C1, gamma=gamma1)
    
    start = time.time()
    #simple_svm.simple_train(small_trainRDD)
    simple_svm.fit(X, y)
    end = time.time()
    print("Simple SVM train on " + str(small_trainRDD.count()) + " samples took " + str(end - start) + " time.")
    print '[Simple Model] number of support vectors = ', len(simple_svm.support_)

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

