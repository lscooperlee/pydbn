## PyDBN: A DBN implementation in Python.


PyDBN is a basic implementation for DBN. It uses the mnist digits recognition data set for testing. With the three layers DBN and 500 neural neurons for each hidden layer, after about 10 minutes training, the accuracy rate could reach 94.5%. It is inspired by a [java implementation](https://github.com/tjake/rbm-dbn-mnist) and [a simple RBM demo in python ](https://github.com/echen/restricted-boltzmann-machines). The algorithm for RBM and DBM implementation comes mainly from *Learning Deep Architectures for AI* by Yoshua Bengio, *A Practical Guide to Training Restricted Boltzmann Machines* by Geoffrey Hinton, *An Introduction to Restricted Boltzmann Machines* by Asja Fischer and Christian Igel, the original *A fast learning algorithm for deep belief nets* by Geoffrey E. Hinton, Simon Osindero and Yee-Whye Teh. Thanks to these guys.


The current version is still a simple implementation, which means the icing on the cake such as momentum and weight-decay is not added yet. Another deference is the [java implementation](https://github.com/tjake/rbm-dbn-mnist) and the algorithm in *A fast learning algorithm for deep belief nets* are both use 2000 hidden units for the top layer. 


In order to run the test code, you need to have python and numpy installed. The code does not include mnist data you have to download yourself.

You could do the following steps to have pyDBN running:

* git clone https://github.com/lscooperlee/pydbn.git

* cd pydbn

* mkdir data

* cd data

* wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

* wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

* wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

* wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

* cd ..

To train a model, simplely run:

* python mnist.py -t training.pickle 	# this will train the DBN, the results are stored in training.pickle

or 

* python mnist.py -t training.json 

to store the model in json format.

test the model by running:

* python mnist.py -r training.pickle   # recognition using the training result stored in training.pickle

A pretrained 4 layer model with 200 units for each of the hidden layers is also provided, this is how you can test it:

* python mnist.py -r models/mnist200hidden.json


cheers
