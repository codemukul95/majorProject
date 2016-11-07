import os

import scipy
from scipy.misc import *

input_path = '/home/mukul/Documents/major/dataset/train/c0'  #enter the path to images folder

img_ext = '.jpg'

os.makedirs('train/c0/')
out_path = os.getcwd()  + '/train/c0/'

for x in os.listdir( input_path ):
    if x.endswith(img_ext):
        img_path = os.path.join(input_path, x)
        img = imread(img_path)
        img_temp = imresize(img, size=(120,160))
        imsave(out_path+x, img_temp)

input_path = '/home/mukul/Documents/major/dataset/train/c1'  #enter the path to images folder

img_ext = '.jpg'

os.makedirs('train/c1/')
out_path = os.getcwd()  + '/train/c1/'

for x in os.listdir( input_path ):
    if x.endswith(img_ext):
        img_path = os.path.join(input_path, x)
        img = imread(img_path)
        img_temp = imresize(img, size=(120,160))
        imsave(out_path+x, img_temp)
