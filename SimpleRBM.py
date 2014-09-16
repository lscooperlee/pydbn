

import numpy as np


class SimpleRBM:
    """
        a RBM
    """

    def __init__(self, num_visible, num_hidden, n_iter=10, learning_rate=0.1):
        # according to JAVA mnist code and echen code, self.__weights are initilized by Gaussian distribution.
        #here we use np.random.normal, np.random.randn is another option
        #also acording to JAVA version mnist code,  we use a standard Gaussian Distribution with mean 0 and deviation 1, however, https://github.com/echen/restricted-boltzmann-machines use deviation 0.1

        self.__weights=np.random.normal(size=(num_visible,num_hidden))
        self.__bias_visible=np.random.normal(size=(1,num_visible))
        self.__bias_hidden=np.random.normal(size=(1,num_hidden))

        self.__num_visible=num_visible
        self.__num_hidden=num_hidden

        self.__learning_rate=learning_rate


    # for training, we use CD1 1batch at the moment.

    # one different between echen and JAVA mnist code is:  echen uses full training, which means in one iteration, it use all training data in one time and calculate the weights.  whereas in JAVA mnist code, it uses batches, which means in one iteration, it runs a for loop, and in the loop only one sample of training data is used to calculate the weights, after the for loop finished, it will update the weights by divided the length of the traning data.

    #however,  according to the Algorithm 1 of  training_restricted_boltzmann_machines_an_introduction, and Algorithm 1 of learning_ensemble_classifiers_via_restricted_boltzmann_machines, they don't divided the weights by the total number of samples in training data. 
    #in these Algorithm 1s, they all have maxmium T (for learning_ensemble_classifiers_via_restricted_boltzmann_machines) or k (for training_restricted_boltzmann_machines_an_introduction) times iteration (V0-H0-V1-H1-V2-H2-V3-H3 ..... ) and don't update weights in between, after that, they will update weights and don't divided by the number of sample

    #here we don't divided ???????????????         because the JAVA code use batch 30, and echen code use all data means batch all, so they need to devide?????????????????????????. 
    #note that in echen code, he use the whole training data as a matrix, or a list of input vectors, so everytime he caculates prob0_hidden, prob1_visible, prob1_hiden he actually change every vectors in this list, than he np.dot the whole list to get weights, because it is like a matrix, so he actually add all input vectors weights together, so that is why here he needs to divide. the same is for JAVA code, they use 30 batches, but they don't keep a list of weights, they just use one weigths, so every time in the loop of batches, they add weights, so they need divided outside the batch loop

    #here we use 1 batch, so we don't devide???????????

    def train(self,data, train_iter=5000):
        for _ in range(train_iter):
            for visible0 in data:
                prob0_hidden_given_visible=self.__visible_hidden_prob(visible0)

                #see a_practical_guide_to_training_restricted_boltzmann_machines Section 3 for detail.
                # accordin to a_practical_guide_to_training_restricted_boltzmann_machines Section 3, for CD1, if the input is logistic function, then use probability no matter driven by data (V0-H0) or in driven by reconstruction (V1-H1).

                # but if the visible units are not logistic function, then when the hidden units are being driven by data (V0-H0), always use stochastic binary states for hidden units, which means we have to add sample for H0, (ie: V0->H0->sample(H0) ), then in reconstruction process (sample(H0)->V1) , we use sample H0 to get V1, but this time, because it is driven by reconstruction, we don't need sample, just (V1-H1)
                #see echen code and JAVA version mnist code. they all have the same process. V0->H0->sample(H0)->V1->H1

                #BUT, learning_ensemble_classifiers_via_restricted_boltzmann_machines seems different, in P164, they use V0-H0->sample(H0)->V1->sample(V1)->H1. also the Algorithm 1 in training_restricted_boltzmann_machines_an_introduction and the Algorithm 1 in chapter 5 of Learning_Deep_Architectures_for_AI all suggest V0-H0->sample(H0)->V1->sample(V1)->H1. MAYBE they use CDn rather than CD1, see a_practical_guide_to_training_restricted_boltzmann_machines Section 3 for details. but the codes use CD1, I think.

                sample_prob0=self.__sample(prob0_hidden_given_visible)

                prob1_visible_given_hidden=self.__hidden_visible_prob(sample_prob0)

                prob1_hidden_given_visible=self.__visible_hidden_prob(prob1_visible_given_hidden)


                # according to a_practical_guide_to_training_restricted_boltzmann_machines Section 3.3, we could use <Pi Hj>data or <Pi Pj>data to calculate the delta W (see equation 9, where  <Vi Hi>data is driven by data, or the first term of equation 9 in training_restricted_boltzmann_machines_an_introduction, which is easy to calculate, <Vi Hj>recon is the second term of equation 9, because p(v,h) in equation 9 is hard to calculate,(because it is a combinition of certain v and certain h, run over all values of the respective variables, which O(2^(m*n)) if for binary value) , that is why gibbs is needed ). in the code, Pi = Vi, Pj is sigmond or probability for hidden layers, Hi is sampled probability. according to a_practical_guide_to_training_restricted_boltzmann_machines Section 3.3. we use <Pi Pj> here.
                self.__weights+=self.__learning_rate*(np.dot(np.matrix(visible0).T, prob0_hidden_given_visible)-np.dot(np.matrix(prob1_visible_given_hidden).T, prob1_hidden_given_visible))

                self.__bias_visible+=visible0-prob1_visible_given_hidden

                self.__bias_hidden+=prob0_hidden_given_visible-prob1_hidden_given_visible

    def __visible_hidden_prob(self,visible_vector):
        h=np.dot(visible_vector,self.__weights)+self.__bias_hidden
        return self.__sigmond(h)

    def __hidden_visible_prob(self, hidden_vector):
        # hidden_vector is sampled vector here, see above comment in train method
        v=np.dot(hidden_vector, self.__weights.T)+self.__bias_visible;
        return self.__sigmond(v)

    #according to JAVA version mnist code, use uniform distribution, echen code uses np.random.rand, which is also an uniform distribution.
    def __sample(self,prob_hidden):
        sample=prob_hidden > np.random.random(self.__num_hidden)
        return sample

    def __sigmond(self,x):
        return 1.0/(1.0+np.exp(-x))

    def __str__(self):
        ret="A {0} class with {1:d} input elements and {2:d} output elements".format(self.__class__,self.__num_visible, self.__num_hidden)
        return ret



if __name__=='__main__':
    s=SimpleRBM(6,2)
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    s.train(training_data)
    print(s)

