#This code used to classify images in a supervised learning setting
#Different classifiers are used namely:
#SVM,

from __future__ import division
from __future__ import print_function
from datetime import datetime
import subprocess
from time import time
from operator import itemgetter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
from StringIO import StringIO
from urlparse import urlparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint
from sklearn.linear_model import RidgeCV
import pandas as pd
import urllib2
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# from multiprocessing.dummy import Pool as ThreadPool #multithreading
#from sklearn import grid_search
#from sklearn.grid_search import GridSearchCV
#from sklearn.grid_search import RandomizedSearchCV
#from sklearn.cross_validation import  cross_val_score

def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
        directory (and all its subdirectories). Symbolic links are ignored.

        Args:
        directory (str): directory to process.

        Returns:
        list of list of float: a list of feature vectors.
        '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

        Args:
        image_path (str): path of the image file to process.

        Returns:
        list of float: feature vector on success, None otherwise.
        '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None


def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.

        Args:
        image (PIL.Image): image to process.
        blocks (int, optional): number of block to subdivide the RGB space into.

        Returns:
        list of float: feature vector if successful. None if the image is not
        RGB.
        '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0] / (256 / blocks))
        gidx = int(pixel[1] / (256 / blocks))
        bidx = int(pixel[2] / (256 / blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x / pixel_count for x in feature]


def train(dir_path, print_metrics=True):
    '''Trains a classifier. dir_path should be a directory paths and should contain subdirectories where each one 
        is for a single class. Then the subdirectories are processed by
        process_directory().

        Args:
        training_path (str): directory containing subdirectories, each subdirectry contains images of a class.
        print_metrics  (boolean, optional): if True, print statistics about classifier performance.

        Returns:
        A classifier e.g. (sklearn.svm.SVC).
        '''

    if not os.path.isdir(dir_path):
        raise IOError('%s is not a directory' % dir_path)

    d = {}
    i = 0
    for subdir, dirs, files in os.walk(dir_path):
        if (i == 0):
            i += 1
            continue
        d["training_{0}".format(i)] = process_directory(os.path.join(subdir))
        i += 1

    # double check the bounndry values
    # data contains all the training data (a list of feature vectors)
    temp = len(d) - 1
    data = d["training_{0}".format(len(d))]
    for x in xrange(temp, 0, -1):
        data = data + d["training_{0}".format(x)]
    #Chnage this to parameters later
    test_size = 0.20
    traing_size = 1 - test_size
    num_trails = 5

    # target is the list of target classes for each feature vector: a '0' for
    # class A and '1' for class B, etc

    target = [len(d)] * len(d["training_{0}".format(len(d))])
    for z in xrange(temp, 0, -1):
        target = target + [z] * len(d["training_{0}".format(z)])


    # split training data in a train set and a test set. The test set will
    # containt 20% of the total
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20)

    # define the parameter search space
    # parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
    #               'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it

    for x in range(0, num_trails):
        f = open("Run_"+str(x)+ "_" + "dir_"+ sys.argv[2] + "_" +str(datetime.now()) +".txt", "w")
        print("Welcome to Classical classification! Run no: %d" %(x))
        f.write("Welcome to Classical classification!")
        print("The working dir is: %s" % (sys.argv[1]))
        f.write("\nThe working dir is: %s\n" % (sys.argv[1]))
        print("The data has been divided with for %.2f percent for training and %.2f percent for testing" % (
            traing_size, test_size))
        print("\n\n========================================================\n\n")
        f.write("The data has been divided with for %.2f percent for training and %.2f percent for testing" % (
            traing_size, test_size))
        f.write("\n\n========================================================\n\n")

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]
        best_score = 0.0
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=3, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB()]
        # iterate over classifiers

        # pool = ThreadPool(4)

        for clf in classifiers:
            print('\n')
            f.write('\n')
            print(clf)
            f.write(str(clf))
            classifier = clf.fit(x_train, y_train)
            results = cross_val_score(classifier, x_test, y_test, cv=5)
            if np.float64(results.mean(dtype=np.float64)).item() > best_score:
                best_score = np.float64(results.mean(dtype=np.float64)).item()
                best_clf = clf
            print('Classifier score')
            f.write('Classifier score')
            print(metrics.classification_report(y_test, classifier.predict(x_test)))
            f.write(metrics.classification_report(y_test, classifier.predict(x_test)))
            print('Confusion Matix: ')
            f.write('Confusion Matix: \n')
            print(metrics.confusion_matrix(y_test, classifier.predict(x_test)))
            f.write(str(metrics.confusion_matrix(y_test, classifier.predict(x_test))))
            print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
            f.write("\nAccuracy: %.3f (%.3f)" % (results.mean(), results.std()))
            print("\n\n========================================================\n\n")
            f.write("\n\n========================================================\n\n")
        #
        print("\n\nBest Obtained Result: %f" % (best_score))
        f.write("\n\nBest Obtained Result: %f" % (best_score))
        print("Best Classifer: %s" % (best_clf))
        f.write("\nBest Classifer: %s" % (best_clf))
        f.close()
        # exit()
        # return classifier

# def single_classifier_training (clf_name):
#     classifier = clf.fit(x_train, y_train)
#     results = cross_val_score(classifier, x_test, y_test, cv=5)
#     metrics.classification_report(y_test, classifier.predict(x_test))
#     metrics.confusion_matrix(y_test, classifier.predict(x_test))
#     "Accuracy: %.3f (%.3f)" % (results.mean(), results.std())


def main(path):
    '''
        Args:
        training_path (str): directory containing subdirectories, each subdirectry contains images of a class.
        '''

    # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    #          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    #          "Naive Bayes", "QDA"]
    # print(*names, sep='\n')
    #classifier = \
    train(path)

def show_usage():
    '''Prints how to use this program
        '''
    print("Error: Missing arg!")
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        show_usage()
    main(sys.argv[1])
