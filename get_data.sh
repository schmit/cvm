#!/usr/bin/env bash

# Clean data
rm -rf data

# Get MNIST data
mkdir -p data/mnist
curl http://www.pjreddie.com/media/files/mnist_train.csv -o data/mnist/mnist_train.csv
curl http://www.pjreddie.com/media/files/mnist_test.csv -o data/mnist/mnist_test.csv

# Get covertype data
# mkdir -p data/covertype
# curl https://archive.ics.uci.edu/ml/machine-learning-databases/covertype/covtype.data.gz -o data/covertype/covtype.data.gz


