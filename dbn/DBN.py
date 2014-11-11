

import numpy as np
import random
import pickle

from .RBM import RBM
from .PCN import PCN


class DBN:


    def __init__(self, *argt):
        self._weights_list=[]
        self._NN_list=[]

        if len(argt)==1 and isinstance(argt[0],int):         #so that one does not have to import DBN, RBM and PCN at the same time.
            for _ in range(argt[0]-1):
                self._NN_list.append(RBM())
            self._NN_list.append(PCN())
        else:
            self._NN_list=list(argt)



    # NOTE THAT: the algorithm to train a DBN seems wrong, see "learning_deep_architectures_for_AI" chapter 6 and java code for detail.
    # in java_mnist notice the stopAt parameter, it consists with the learning_deep_architectures_for_AI algorithm.

    # This algorithm is wrong, according to learning_deep_architectures_for_AI, chapter 6, the input data for a RBM is generated everytime from the raw input in each iteration. Note that java mnist code does the same thing: mnist/BinaryMinstDBN.java --> learn method shows in every iteration, a batch (30 item) is chose to be the input for trainer.learn, which is a StackedRBMTrainer.java --> learn method. in StackedRBMTrainer, the learn method generate the input from raw input data layer after layer until the current one, then inputTrainer.learn is called, which is SimpleRBMTrainer.java --> learn method. It implements the RBMUpdate algorithm in learning_deep_architectures_for_AI, which update all weights and biases only once.

    def train(self, data, out, num_hidden=4, train_iter=5000, learning_rate=0.01):

        # if self._NN_list == 1, that should be a PCN, #so one can still use 1-layer DBN as a PCN without even import it
        if len(self._NN_list) < 1:
            raise Exception

        if not isinstance(self._NN_list[-1], PCN):
            raise Exception

        #make the training data and output random, good for sequenced data.
        #NO the result is very bad, is suffling bad for training??
#        tmp=[ x for x in zip(data,out) ]
#        np.random.shuffle(tmp)
#        tmp=[ x for x in zip(*tmp) ]
#        in_data=tmp[0]
#        out_data=tmp[1]

        in_data=data
        out_data=out

        print('start training DBN with {0} input, {1} hidden layer and {2} output'.format(len(in_data), num_hidden, len(out_data)))

#        for odr, rbm in enumerate(self._NN_list[:-1]):
#            print('training the {0}th layer of DBN: {1}'.format(odr+1,rbm))
#            rbm.train(in_data, num_hidden, train_iter, learning_rate)
#            
#            in_data=data
#            for i in range(odr+1):
#                trained_rbm=self._NN_list[i]
#                in_data=trained_rbm.forward_data(in_data)
#
#        pcn=self._NN_list[-1]
#        print('training the PCN layer')
#        pcn.train(in_data, out, train_iter, learning_rate)


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


#        pcn=self._NN_list[-1]
#        print('training the PCN layer')
#        for trained_rbm in self._NN_list[:-1]:
#            in_data=trained_rbm.forward_data(in_data)
#        pcn.train(in_data, out, train_iter, learning_rate)

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

