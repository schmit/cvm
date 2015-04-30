'''
Some plotting functions for SVMs
'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

def plot_2d_model(model, X, y, **kwargs):
    '''
    Visualizes a model on 2d dataset (X, y)

    model should support:
        . fit
        . predict
        . decision_function
        . support_vectors_
        . n_support_
    '''
    # fit model
    model.fit(X, y)

    # predict and compute roc auc
    if 'X_test' in kwargs and 'y_test' in kwargs:
        X_test = kwargs['X_test']
        y_test = kwargs['y_test']
    else:
        X_test = X
        y_test = y

    y_pred = model.predict(X_test)
    rocauc = roc_auc_score(y_test, y_pred)

    # plot results
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c='k',
               s=25, zorder=10, cmap=plt.cm.Paired, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y,
               s=5, alpha=0.8, zorder=5, cmap=plt.cm.Paired)

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # put the result into a color plot
    Z = Z.reshape(XX.shape)
    ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, alpha=0.1)
    ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5], alpha=0.7)

    # set plot stuff
    ax.set_title('{} support vectors - {:.2f} ROC'.format(np.sum(model.n_support_), rocauc))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

