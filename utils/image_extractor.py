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



class input_sample:
    def __init__(self, labelname, pixels):
        self.label = labelname
        self.pixels = pixels



class image_extractor:
    def __init__(self):
        return
    
    def extractImages(self, path, xres, yres, maxFiles):
        inputs = []
        count = 0
        for filename in os.listdir(path):
            stripedName = filename.split('.')[0]

            if (stripedName == 'cat'):
                label = 0
            else:
                label = 1
            jpg = Image.open(path+'/'+filename)
            resizedjpg = jpg.resize((xres, yres), Image.ANTIALIAS)
            jpgarr = [r for r, g, b, in resizedjpg.getdata()] + [g for r, g, b, in resizedjpg.getdata()] + [b for r, g, b, in resizedjpg.getdata()]
            pixels = jpgarr #image.extract_pixels_2d(jpgarr, (2,2))
            currentSample = input_sample(label, pixels)
            inputs.append(currentSample)

            if count > maxFiles:
                break
            count += 1
        return inputs

