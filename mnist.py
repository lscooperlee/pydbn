import os
import numpy as np
import struct, gzip

from dbn.DBN import DBN
from dbn.RBM import RBM
from dbn.PCN import PCN

class mnist():

    def __init__(self):
        self.train_fname_images = 'data/train-images-idx3-ubyte.gz'
        self.train_fname_labels = 'data/train-labels-idx1-ubyte.gz'
        self.test_fname_images  = 'data/t10k-images-idx3-ubyte.gz'
        self.test_fname_labels  = 'data/t10k-labels-idx1-ubyte.gz'

    def _read_labels(self, fname):
        f = gzip.GzipFile(fname, 'rb')
        magic_nr, n_examples = struct.unpack(">II", f.read(8))
        labels = np.fromstring(f.read(), dtype='uint8').reshape(n_examples, 1)
        return labels

    def _read_images(self, fname):
        f = gzip.GzipFile(fname, 'rb')
        magic_nr, n_examples, rows, cols = struct.unpack(">IIII", f.read(16))
        shape = (n_examples, rows*cols)
        images = np.fromstring(f.read(), dtype='uint8').reshape(shape)
        return images

    def data(self,group='train'):

        if 'train' == group:
            images = self._read_images(self.train_fname_images)#[:100]
            labels = self._read_labels(self.train_fname_labels)#[:100]
        elif 'test' == group:
            images = self._read_images(self.test_fname_images)
            labels = self._read_labels(self.test_fname_labels)
        else:
            raise Exception

        return images/255, labels

    @classmethod
    def printMnist(cls, data):
        for n,d in enumerate(data):
            print(d,end=" ")
            if not ((n+1) % 28):
                print("")



if __name__=='__main__':
    import sys
    import argparse
    from collections import Counter
    db=mnist()
    dbn=RBM()*3+PCN()

    parser=argparse.ArgumentParser()
    parser.add_argument('-t', metavar="ModelFile", action="store", nargs=1, type=str, help="model file to save, support both pickle and json format.")
    parser.add_argument('-r', metavar="ModelFile", action="store", nargs=1, type=str, help="model file used in recognition")

    args=parser.parse_args(sys.argv[1:])
    

    if args.t:
        filename=args.t[0]
        train_image, train_labels=db.data('train')

        lst=[]
        for i in train_labels:
            z=np.zeros(10)
            z[i[0]]=1
            lst.append(z)
        
        train_labels=np.array(lst)
        dbn.train(train_image, train_labels, num_hidden=500, train_iter=30000,learning_rate=0.1)

        if filename.endswith('json'):
            dbn.savejson(filename)
        else:
            dbn.save(filename)

    if args.r:
        filename=args.r[0]
        test_image, test_labels=db.data('test')
        if filename.endswith('json'):
            dbn.loadjson(filename)
        else:
            dbn.load(filename)

        retlst=[]
        for i,j in zip(test_image,test_labels):
            r=dbn.forward(i)
            retlst.append(np.argmax(r)==j[0])
        c=Counter(retlst)
        print("accuracy: {:.3f}".format(c[True]/len(retlst)))


    

