# CVM (Work in progress)

A PySpark package that that implements Kernel SVM classification based on the Cascading SVM idea presented in [1].

Note that currently the implementation only runs on single machines (though on multiple cores), as we are not distributing the model object for prediction.

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

To obtain the data, run the `./get_data.sh` script.

### MNIST Digit classification

In this example, we classify the handwritten digit.

From the top level directory, run
```
[SPARKDIR]/bin/spark-submit --driver-memory 2g bin/mnist.py
```

### Forest Covertype classification

In this example, we try to classify whether the forest covertype is of type 2 or not, such that the problem becomes a binary classification problem.

From the top level directory, run
```
[SPARKDIR]/bin/spark-submit --driver-memory 2g bin/covertype.py
```

## Authors

- Carlos Riquelme
- Lan Huong
- Sven Schmit

## References

[1] Graf, Hans P., Eric Cosatto, Leon Bottou, Igor Dourdanovic, and Vladimir Vapnik. "Parallel support vector machines: The cascade svm." In *Advances in neural information processing systems*, pp. 521-528. 2004.
