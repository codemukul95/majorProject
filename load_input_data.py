import os
import scipy
from scipy.misc import *
import numpy as np
from tflearn.data_utils import shuffle

def load_data():
    X_train = []
    Y_train = []
    img_ext = '.jpg'

    #reading c0 first
    input_path = "/home/mukul/Documents/major/dataset/data/train/c0/"
    i = 0
    for x in os.listdir( input_path ):
        if x.endswith( img_ext ) and i<2000 :
            img_path = os.path.join(input_path, x)
            img = imread(img_path)
            img_float64 = np.float64(img)
            X_train.append(img_float64)
            Y_train.append(1)
            i += 1
            print i

    #reading c1 now
    input_path = "/home/mukul/Documents/major/dataset/data/train/c1/"
    i = 0
    for x in os.listdir( input_path ):
        if x.endswith( img_ext ) and i<2000:
            img_path = os.path.join(input_path, x)
            img = imread(img_path)
            img_float64 = np.float64(img)
            X_train.append(img_float64)
            Y_train.append(0)
            i += 1
            print i

    return (X_train,Y_train)
