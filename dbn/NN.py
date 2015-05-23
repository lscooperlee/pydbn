
import numpy as np
import pickle
import json

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

    def savejson(self, filename):
        with open(filename, 'w') as fd:
            newdict={}
            param=self.get_param()
            for i in param.keys():
                newdict[i]=np.round(param[i],decimals=3).tolist()

            json.dump(newdict,fd)


    def __str__(self):
        ret="A {0} class ".format(self.__class__)
        return ret


