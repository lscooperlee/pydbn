

import numpy as np
import random
from .NN import NN

class RBM(NN):

    def __init__(self):
        self._weights=None
        self._bias_visible=None
        self._bias_hidden=None

    def train(self, data, num_hidden=4, train_iter=5000, learning_rate=0.1):
        num_visible=np.shape(data)[1]
        self.init_params(num_visible, num_hidden)
        self.__train_1step_1batch_basic(data, train_iter, learning_rate)

    def set_param(self, param):
        self._weights=param['weights']
        self._bias_visible=param['bias_visible']
        self._bias_hidden=param['bias_hidden']

    def get_param(self):
        param={ 'weights':self._weights, 'bias_visible':self._bias_visible, 'bias_hidden':self._bias_hidden }
        return param

    def forward_data(self, data):
        """
        this function is used to forward the whole dataset, not just one vector in the dataset, thus it loops all vectors in the dataset and it uses self.forward function to forward every vector.
        this function is basically used in DBN training, when getting the whole inputset for the next hidden layer.
        """
        out_array=[]
        for i in data:
            out_array.append(self.forward(i).reshape(-1))    # it is a [[num,num]] format, which is a matrix or 2d array
        return np.array(out_array)


    def forward(self,visible_vector):
        """
        it is assumed the visible_vector is a matrix or 2d array
        """
        return self.__visible_hidden_prob(visible_vector)


    def init_params(self, num_visible, num_hidden):
        self._weights=np.random.normal(scale=0.1, size=(num_visible,num_hidden))
        self._bias_visible=np.random.normal(scale=0.1, size=(1,num_visible))
        self._bias_hidden=np.random.normal(scale=0.1, size=(1,num_hidden))


    def iter_update(self, dinput, learning_rate, **kwargs):
        self.__rbm_update(dinput, learning_rate)


    def __train_original(self,data, train_iter=5000, learning_rate=0.1):
        '''
        the standard 1-step training method with full batch (over the full data in one iteration),without any other speeding up tricks, just list as a reference, don't use it because it is very time-consuming.
        '''
        for _ in range(train_iter):
            for _data in data:
                self.__rbm_update(_data, learning_rate)

            # we add all weights, bias_visible and bias_hidden over all data in one iteration, so have to average it for this round.
            self._weights=self._weights/len(data)
            self._bias_visible=self._bias_visible/len(data)
            self._bias_hidden=self._bias_hidden/len(data)


    def __train_1step_1batch_basic(self,data, train_iter=5000, learning_rate=0.1):
        '''
        this is the fixed 1 step 1 batch basic train method without momentum or other ways to speed up.
        '''
        for _ in range(train_iter):

                _data = random.choice(data)

                self.__rbm_update(_data, learning_rate)



    def __rbm_update(self, _data, learning_rate):

        visible0=np.matrix(_data)           # change 1d array to 2d array, namely matrix

        prob0_hidden_given_visible=self.__visible_hidden_prob(visible0)

        sample_prob0=self.__sample(prob0_hidden_given_visible)

        prob1_visible_given_hidden=self.__hidden_visible_prob(sample_prob0)

        prob1_hidden_given_visible=self.__visible_hidden_prob(prob1_visible_given_hidden)

        self._weights+=learning_rate*(np.dot(visible0.T, prob0_hidden_given_visible)-np.dot(prob1_visible_given_hidden.T, prob1_hidden_given_visible))

        #bias_visible and bias_hidden should be multipled by learning_rate too
        self._bias_visible+=learning_rate*(visible0-prob1_visible_given_hidden)

        self._bias_hidden+=learning_rate*(prob0_hidden_given_visible-prob1_hidden_given_visible)


    def __visible_hidden_prob(self,visible_vector):
        h=np.dot(visible_vector,self._weights) + self._bias_hidden
        return self.sigmond(h)

    def __hidden_visible_prob(self, hidden_vector):
        v=np.dot(hidden_vector, self._weights.T)+self._bias_visible;
        return self.sigmond(v)

    def __sample(self,prob_hidden):
        sample=prob_hidden > np.random.random(len(prob_hidden))
        return sample

    def __str__(self):
        ret="A {0} class".format(self.__class__)
        return ret

    def __add__(self, cls):
        from .DBN import DBN
        if self == cls:
            cls=self.__class__()
        return DBN(self,cls)

    def __mul__(self, num):
        DBN=self
        if num <= 0:
            raise Exception
        for i in range(num-1):
            DBN+=self.__class__()
        return DBN
