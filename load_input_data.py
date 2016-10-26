import os
import scipy
from scipy.misc import *



def load_data():
    X_train = []
    Y_train = []
    img_ext = '.jpg'

    #reading c0 first
    input_path = "/path/to/resize/train/c0"
    for x in os.listdir( input_path ):
        if x.endswith( img_ext ):
            img_path = os.path.join(input_path, x)
            img = imread(img_path)
            X_train.append(img)
            Y_train.append(1)

    #reading c1 now
    input_path = "path/to/resize/train/c1"
    for x in os.listdir( input_path ):
        if x.endswith( img_ext ):
            img_path = os.path.join(input_path, x)
            img = imread(img_path)
            X_train.append(img)
            Y_train.append(0)

    return (X_train,Y_train)
            
