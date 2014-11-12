

import numpy as np
import random
import pickle

from .RBM import RBM
from .PCN import PCN


class DBN:


    def __init__(self, *argt):
        self._weights_list=[]
        self._NN_list=[]

        if len(argt)==1 and isinstance(argt[0],int): 
            for _ in range(argt[0]-1):
                self._NN_list.append(RBM())
            self._NN_list.append(PCN())
        else:
            self._NN_list=list(argt)



    def train(self, data, out, num_hidden=4, train_iter=5000, learning_rate=0.01):

        if len(self._NN_list) < 1:
            raise Exception

        if not isinstance(self._NN_list[-1], PCN):
            raise Exception

        in_data=data
        out_data=out

        print('start training DBN with {0} input, {1} hidden layer and {2} output'.format(len(in_data), num_hidden, len(out_data)))

        num_visible=len(in_data[0])
        for odr, nn in enumerate(self._NN_list[:-1]):
            nn.init_params(num_visible,num_hidden)
            num_visible=num_hidden
        nn=self._NN_list[-1]
        nn.init_params(num_visible,len(out_data[0]))

        for odr, nn in enumerate(self._NN_list):
            for _ in range(train_iter):

                _i = random.randrange(len(in_data))
                _dinput=in_data[_i]
                _doutput = out_data[_i]

                for i in range(1,odr+1):
                    trained_rbm=self._NN_list[i-1]
                    _dinput = trained_rbm.forward(_dinput)

                kwargs={'dinput':_dinput, 'doutput':_doutput, 'learning_rate': learning_rate}
                nn.iter_update(**kwargs)


    def save(self, filename):
        with open(filename,'wb') as fd:
            for nn in self._NN_list:
                pickle.dump(nn.get_param(),fd)


    def load(self, filename):
        with open(filename,'rb') as fd:
            for nn in self._NN_list:
                nn.set_param(pickle.load(fd))


    def forward(self, input_vector):
        i=input_vector
        for rbm in self._NN_list[:-1]:
            i=rbm.forward(i)

        pcn=self._NN_list[-1]
        o=pcn.forward(i)

        return o

    def __add__(self,cls):
        if cls in self._NN_list and isinstance(cls, RBM):
            cls=RBM()
        if cls in self._NN_list and isinstance(cls, PCN):
            raise Exception

        self._NN_list.append(cls)
        return self
        
    def __str__(self):
        ret="A {0} class {1}".format(self.__class__,self._NN_list)
        return ret

    def __len__(self):
        return len(self._NN_list)

