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
        # 2 big-ending integers
        magic_nr, n_examples = struct.unpack(">II", f.read(8))
        # the rest, using an uint8 dataformat (endian-less)
        labels = np.fromstring(f.read(), dtype='uint8').reshape(n_examples, 1)
        return labels

    def _read_images(self, fname):
        f = gzip.GzipFile(fname, 'rb')
        # 4 big-ending integers
        magic_nr, n_examples, rows, cols = struct.unpack(">IIII", f.read(16))
        shape = (n_examples, rows*cols)
        # the rest, using an uint8 dataformat (endian-less)
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

        return images, labels


if __name__=='__main__':
    import sys
    db=mnist()
    #construct a 3 layer DBN
    dbn=RBM()*2+PCN()
    try:
        if sys.argv[1] == '-t':
            filename=sys.argv[2]
            train_image, train_labels=db.data('train')
            train_image=train_image / 255

            lst=[]
            for i in train_labels:
                z=np.zeros(10)
                z[i[0]]=1
                lst.append(z)
            
            train_labels=np.array(lst)
            dbn.train(train_image, train_labels, num_hidden=500, train_iter=30000,learning_rate=0.1)
            dbn.save(filename)

        elif sys.argv[1] == '-r':
            filename=sys.argv[2]
            test_image, test_labels=db.data('test')
            test_image=test_image / 255
            dbn.load(filename)
            total=0
            correct=0
            for i,j in zip(test_image,test_labels):
                r=dbn.forward(i)
                r=np.argmax(r)
                total+=1
                if r == j[0]:
                    correct+=1
            print(correct/total)

        elif sys.argv[1]=='-o':
            pcn=DBN(1)
            train_image, train_labels=db.data('train')
            train_image=train_image/255

            lst=[]
            for i in train_labels:
                z=np.zeros(10)
                z[i[0]]=1
                lst.append(z)
            
            train_labels=np.array(lst)
            pcn.train(train_image, train_labels, train_iter=30000,learning_rate=0.1)

            test_image, test_labels=db.data('test')
            test_image=test_image/255
            total=0
            correct=0
            for i,j in zip(test_image,test_labels):
                r=pcn.forward(i)
                r=np.argmax(r)
                total+=1
                if r == j[0]:
                    correct+=1
            print(correct/total)

        else:
            print('usage:\n{0} -t picklefile\t\tfor train\n{0} -r picklefile \t\tfor recognition\n'.format(sys.argv[0]))
            sys.exit()
    except Exception as e:
        print(e)
        print('usage:\n{0} -t picklefile\t\tfor train\n{0} -r picklefile \t\tfor recognition\n'.format(sys.argv[0]))
        sys.exit()
        

