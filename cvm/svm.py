'''
WIP Combines Cascade and Models
'''

from cascade import Cascade
from models import SVMModel

class KernelSVM:
    def __init__(self, nmax):
        model = SVMModel()
        self.cascade = Cascade(model, nmax)

    def train(self, labeledPoints):
        self.cascade.train(labeledPoints)

    def predict(self, features):
        return self.cascade.predict(features)