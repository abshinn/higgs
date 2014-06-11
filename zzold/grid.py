#!/usr/bin/env python2.7
"""running grid search on higgs boson data..."""

from pprint import pprint
from time import time
import logging

from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
# import pandas as pd
import numpy as np
# import json

def run_grid(X, y):
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    pipeline = Pipeline([
#     ('rdge', RidgeClassifier()),
#     ( 'kNN', KNeighborsClassifier()),
        ( 'clf', LogisticRegression()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
#     'rdge__alpha': (0.01, 0.1, 1, 10),
        'clf__C': (0.1, 1, 10),
        'clf__penalty': ('l1', 'l2'),
    }

    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.1, random_state=42)

    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring="roc_auc")

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(xtrain, ytrain)
    print("done in %0.3fs\n" % (time() - t0))

    print("all scores:")
    for item in grid_search.grid_scores_:
        print item

    print("\nBest score: %0.3f" % grid_search.best_score_)
    print("\nBest parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
if __name__ == "__main__":
    print(__doc__)

    # load data
    X = np.load("data/X_train.npy")
    y = np.load("data/y_train.npy")
#    weight = np.load("data/weight.npy")
    
    print("loaded X, shape: {}".format(X.shape))
     
    na_ind = np.where(X == -999.)
    X_na = X[na_ind[0]]
    y_na = y[na_ind[0]]
    X = np.delete(X, na_ind[0], axis = 0)
    y = np.delete(y, na_ind[0], axis = 0)

    print("\n### without NA ###")
    print("training set shape: {}".format(X.shape))
    run_grid(X, y)

#     print("\n### with NA ###")
#     run_grid(X_na, y_na)
