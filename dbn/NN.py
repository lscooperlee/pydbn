
import numpy as np
import pickle

class NN:

    def __init__(self):
        pass

    def sigmond(self,x):
        return 1.0/(1.0+np.exp(-x))

    def save(self, filename):
        with open(filename,'ab') as fd:
            pickle.dump(self.get_param(),fd)

    def load(self, filename):
        with open(filename, 'rb') as fd:
            self.set_param(pickle.load(fd))

    def __str__(self):
        ret="A {0} class ".format(self.__class__)
        return ret


