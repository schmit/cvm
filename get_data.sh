#!/usr/bin/env bash

# Clean data
rm -rf data

# Get MNIST data
mkdir -p data/mnist
curl http://www.pjreddie.com/media/files/mnist_train.csv -o data/mnist/mnist_train.csv
curl http://www.pjreddie.com/media/files/mnist_test.csv -o data/mnist/mnist_test.csv

# Get covertype data
mkdir -p data/covertype
curl http://stanford.edu/~schmit/misc/covtype/train.data -o data/covertype/train.data
curl http://stanford.edu/~schmit/misc/covtype/test.data -o data/covertype/test.data
