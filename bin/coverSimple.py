import sys

from cvm.svm import NuSVC, SVC
from sklearn.cross_validation import train_test_split

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint


def filter2classes(line):
	words = line.split(",")
	return (int(words[-1]) < 3)

def parseData(line):
    fields = line.strip().split(",")
    label = int(fields[-1])
    feature = [float(feature) for feature in fields[0: 9]] #fields[0: -1]
    return LabeledPoint(label, feature)


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
    data = sc.textFile('data/covertype/scaled_covtype.data').collect()
    
    # Subsample and split data aproximately into training (60%) and test (40%)
    data, throwaway = train_test_split(data, train_size = 0.1)    
    small_train, small_test = train_test_split(data, train_size = 0.6)    

    small_trainRDD = sc.parallelize(small_train).filter(filter2classes).map(parseData).cache()
    small_testRDD = sc.parallelize(small_test).filter(filter2classes).map(parseData).cache()

    print "Size of downsampled train set: ", small_trainRDD.count()
    print "Size of downsapled test set: ", small_testRDD.count()

    print 'Fitting simple model'
    simple_svm = SVC(C = 128.0, gamma=8.0)
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