
import os
from PIL import Image
import glob
import numpy as np
import random
import argparse
import sys
from scipy.ndimage import filters


from dbn.DBN import DBN
from dbn.RBM import RBM
from dbn.PCN import PCN


DATAPATH=os.path.dirname(__file__)
DATAPATH=DATAPATH+"/" if DATAPATH else ""
DATAPATH=DATAPATH+"data/dataset/"

#outputdir='/tmp/dir/'

def load_file():
    imglst=glob.glob(DATAPATH+"*.jpg")

    random.shuffle(imglst)
    lst=[]
    cls=[]

    for n,i in enumerate(imglst):

        with Image.open(i) as img:
            filename=os.path.basename(i)
            imgr=img.resize((96,96)).convert('L')
            r=np.array(imgr)

            imx = np.zeros(r.shape)
            filters.sobel(r, 1, imx)
            imy = np.zeros(r.shape)
            filters.sobel(r, 0, imy)
            mag = 255-np.sqrt(imx**2 + imy**2)

            d=np.array(mag).reshape(-1)

#            r.save(outputdir+i)
            if filename.startswith("dog"):
                lst.append(d)
                cls.append([1,0])
            elif filename.startswith("cat"):
                lst.append(d)
                cls.append([0,1])
    
    return np.array(lst),np.array(cls)

def load_dogcat():
    lst,cls=load_file()

    train_data=lst[:80]/255
    train_label=cls[:80]

    test_data=lst[80:]/255
    test_label=cls[80:]

    return train_data, train_label, test_data, test_label
    



if __name__=='__main__':
    
    argp=argparse.ArgumentParser()
    argp.add_argument("-t", action='store_true')
    argp.add_argument("-r", action='store_true')
    p=argp.parse_args(sys.argv[1:])


    train_data, train_label, test_data, test_label=load_dogcat()

    dbn=RBM()*2+PCN()

    if vars(p)['t']:
        dbn.train(train_data, train_label, num_hidden=1000, train_iter=30000,learning_rate=0.1)
        dbn.save("/tmp/dogcat.pickle")

    elif vars(p)['r']:
        dbn.load("/tmp/dogcat.pickle")

        total=0
        correct=0
        for i,j in zip(train_data,train_label):
            r=dbn.forward(i)
            print(r)
            total+=1
            if np.argmax(r) == np.argmax(j):
                correct+=1
        print(correct/total)
