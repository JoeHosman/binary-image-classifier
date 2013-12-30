# Author: Sudnya Padalikar
# Date  : Dec 29 2013
# Brief : A python script to read the jpg image data from Kaggle dataset into matrix (as input for scikit) 
# Comment : I will try to abide by https://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Python_Style_Rules

#!/usr/bin/python
import os

import numpy
from numpy import array

from PIL import Image
from sklearn.feature_extraction import image

maxFiles = 10

class input_sample:
    def __init__(self, labelname, patches):
        self.label = labelname
        self.patches = patches

class image_extractor:
    def __init__(self):
        return
    
    def extractImages(self, path):
        inputs = []
        count = 0
        for filename in os.listdir(path):
            if (filename.rstrip('.')[0] == 'cat'):
                label = 0
            else:
                label = 1
            jpg = Image.open(path+'/'+filename)
            resizedjpg = jpg.resize((256, 256), Image.ANTIALIAS)
            jpgarr = array(jpg)
            patches = image.extract_patches_2d(jpgarr, (2,2))
            currentSample = input_sample(label, patches)
            inputs.append(currentSample)

            if count > maxFiles:
                break
            count += 1
        return inputs

