# Author: Sudnya Padalikar
# Date  : Dec 29 2013
# Brief : A python script to call image extractor, classifiers etc

#!/usr/bin/python
import argparse

from classifiers.animal_classifier import animal_classifier

from numpy import array

from sklearn import svm
from sklearn import cluster
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from utils.image_extractor import image_extractor
from utils.image_extractor import input_sample

max_random_trees = 255
total_features = 30

def getXY(samples):
    X = []
    Y = []
    for entry in samples:
        Y.append(entry.label)
        #X.append(entry.pixels)
        X.append(entry.pixels)

    return (array(X), array(Y))

def computeAccuracy(predicted, reference):
    correct = 0
    total   = 0

    for p, r in zip(predicted, reference):
        total += 1
        if p == r:
            correct += 1
        
    return (correct * 100.0) / total

def cats_dogs_classifier(path, x, y, trainingfiles):
    random_forest1 = RandomForestClassifier(n_estimators=max_random_trees, max_features=total_features, n_jobs=4)
    random_forest2 = RandomForestClassifier(n_estimators=max_random_trees, max_features=total_features, n_jobs=4)
    gnb = GaussianNB()    
    pipe = Pipeline(steps=[('random_forest1', random_forest1), ('random_forest2', random_forest2), ('gnb', gnb)])
    
    extractor = image_extractor()
    samples = extractor.extractImages(path, x, y, trainingfiles)
    (X, Y) = getXY(samples)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    clf = random_forest1
    clf = clf.fit(X_train, y_train)
    clf = random_forest2
    clf = clf.fit(X_train, y_train)
    clf = gnb
    clf = clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print "Accuracy : ", computeAccuracy(prediction, y_test)
    

def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-path',        help="path to directory containing images to extract", type=str)
    parser.add_argument('-x',           help="x resolution", type=int, default=48)
    parser.add_argument('-y',           help="y resolution", type=int, default=48)
    parser.add_argument('-trainingsize',help="number of training files to use", type=int, default=5)
    args = parser.parse_args()
    cats_dogs_classifier(args.path, args.x, args.y, args.trainingsize)

if __name__ == '__main__':
    main()


