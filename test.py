
import numpy as np

from dbn.DBN import DBN
from dbn.RBM import RBM
from dbn.PCN import PCN


def test_dbn():
    c1=RBM()
    c2=PCN()
    c=c1+c1+c2
    c=DBN(RBM(),RBM(),PCN())
    c=DBN(4)
#    c=DBN(0)
#    c=DBN(PCN())
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    output_data=np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])
    c.train(training_data,output_data,train_iter=5)
    c.save('/tmp/dbntest.pickle')
    c.load('/tmp/dbntest.pickle')
    test_data=np.array([1,0,0,0,0,0])
    d=c.forward(test_data)
    print(d)
    c.savejson('/tmp/dbntest.json')


def test_pcn():
    s=PCN()
    training_data=np.array([[1,1,1],[1,0,1],[0,1,1],[0,1,0]])
    output=np.array([[0,1],[0,1],[1,0],[1,0]])   
    s.train(training_data,output,train_iter=5)
    print(s)
    print(s.get_param())


def test_rbm():
    s=RBM()
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    s.train(training_data,train_iter=5)
    print(s)
    print(s.get_param())
    s.savejson('/tmp/tmp.json')



if __name__=='__main__':
    test_dbn()
    test_pcn()
    test_rbm()
