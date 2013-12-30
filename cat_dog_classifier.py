# Author: Sudnya Padalikar
# Date  : Dec 29 2013
# Brief : A python script to call image extractor, classifiers etc

#!/usr/bin/python
import sys
import os
import argparse


from classifiers.animal_classifier import animal_classifier
from utils.image_extractor import image_extractor
from utils.image_extractor import input_sample

def cats_dogs_classifier(path):
    extractor = image_extractor()
    samples = extractor.extractImages(path)
    #print samples[0].label

def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-path',    help="path to directory containing images to extract", type=str)
    args = parser.parse_args()
    cats_dogs_classifier(args.path)

if __name__ == '__main__':
    main()


