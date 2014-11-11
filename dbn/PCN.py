

import numpy as np
import random

from .NN import NN

class PCN(NN):

    def __init__(self):
        self._weights=None
        self._bias=None

    def train(self, data, output, train_iter=5000, learning_rate=0.1):
        num_output=np.shape(output)[1]
        num_input=np.shape(data)[1]

        self.init_params(num_input,num_output)
        self.__train_random_gradient_descent(data, output, train_iter, learning_rate)


    def set_param(self, param):
        self._weights=param['weights']
        self._bias=param['bias']

    def get_param(self):
        param={'weights':self._weights, 'bias':self._bias}
        return param

    def forward(self, input_vector):
        o=np.dot(input_vector, self._weights)+self._bias
        return o

    def init_params(self, num_input, num_output):
        self._weights=np.random.normal(size=(num_input,num_output))
        self._bias=np.random.normal(size=(1,num_output))

    def iter_update(self,dinput, doutput, learning_rate):
        self.__pcn_update(dinput, doutput, learning_rate)

    def __train_gradient_descent(self, data, output, train_iter=5000, learning_rate=0.1):
        """
        common gradient descent algorithm
        """
        d_o=tuple(zip(data,output))
        for _ in range(train_iter):
            for i,o in d_o:
                self.__pcn_update(i,o,learning_rate)

            #because we add weights for all data set, this leaves a len(data) times weights, so we have to average it.
            self._weights=self._weights/len(data)

    def __train_random_gradient_descent(self, data, output, train_iter=5000, learning_rate=0.1):
        """
        random gradient descent algorithm, choose a random value from data each time, 
        take weight from this data as the weight gradient descent direction of this round iteration.
        """
        d_o=tuple(zip(data,output))
        for _ in range(train_iter):
            i,o=random.choice(d_o)
            self.__pcn_update(i,o,learning_rate);

    
    def __pcn_update(self, dinput, doutput, learning_rate):
        o=np.matrix(doutput)
        i=np.matrix(dinput)
        yi=self.sigmond(np.dot(i,self._weights)+self._bias)
        self._weights+=learning_rate*np.dot(i.T, (o-yi))


    def __str__(self):
        ret="A {0} class ".format(self.__class__)
        return ret

    def __radd__(self, cls):
        from .DBN import DBN
        return DBN(cls, self)

