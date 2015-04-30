'''
Functions to create simple artificial datasets
'''

import numpy as np
import pandas as pd

from math import pi

def spiral_data(n, sigma=0.5, frac=0.5, a=1, b=10, w=1):
    '''
    Generates two spirals
    '''
    y = 2*(np.random.rand(n) < frac)-1
    t = np.random.rand(n) * (b - a) + a
    x1 = y * t * np.cos(w * t) + 0.5 * y + sigma * np.random.randn(n)
    x2 = y * t * np.sin(w * t) + sigma * np.random.randn(n)
    return pd.DataFrame(np.vstack((x1, x2, y)).T, columns=['x1', 'x2', 'y'])

def linear_data(n, sigma=0.5, a=1, b=-0.5):
    '''
    Generates data according to model:
    y = sign(a * x1 + b * x2 + sigma * e)
    '''
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.sign(a * x1 + b * x2 + sigma * np.random.randn(n))
    return pd.DataFrame(np.vstack((x1, x2, y)).T, columns=['x1', 'x2', 'y'])

def circle_data(n, sigma=0.5, frac=0.5, a=1, b=2):
    '''
    Generates a small and big circle
    '''
    y = 2*(np.random.rand(n) < frac)-1
    # radius
    r = np.random.rand(n) * a + b * (y==1)
    # angle
    theta = 2 * pi * np.random.rand(n)
    x1 = np.cos(theta) * r + sigma * np.random.rand(n)
    x2 = np.sin(theta) * r + sigma * np.random.rand(n)
    return pd.DataFrame(np.vstack((x1, x2, y)).T, columns=['x1', 'x2', 'y'])
