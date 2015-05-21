# CVM (Work in progress)

A PySpark package that that implements Kernel SVM classification based on the Cascading SVM idea presented in [1].

## Install

To install the `cvm` package, simply run

```
pip install .
```

To install the `cvm` package in development mode, run

```
pip install -e .
```

## Examples

### MNIST Digit classification

In this example, we classify digits as being 4 or less, versus 5 or higher,
such that it is a binary classification problem.

First download the MNIST dataset in csv format [here](http://www.pjreddie.com/projects/mnist-in-csv/) and place in `data/mnist/` folder

Then, from the top level directory, run
```
[SPARKDIR]/bin/spark-submit --driver-memory 2g bin/mnist.py
```

## Authors

- Carlos Riquelme
- Lan Huong
- Sven Schmit

## References

[1] Graf, Hans P., Eric Cosatto, Leon Bottou, Igor Dourdanovic, and Vladimir Vapnik. "Parallel support vector machines: The cascade svm." In *Advances in neural information processing systems*, pp. 521-528. 2004.
