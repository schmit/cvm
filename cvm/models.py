
import random

class RandomModel:
    def reduce(self, iterator):
        for elem in iterator:
            if random.random() > 0.5:
                yield elem

    def train(self, labeledPoints):
        print [x.label for x in labeledPoints]

    def predict(self, features):
        return 0

class SVMModel:
    def reduce(self, iterator):
        pass

    def train(self, labeledPoints):
        pass

    def predict(self, features):
        pass