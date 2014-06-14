#!/usr/bin/env python2.7
"""train stumbleupon..."""

# from data_scrub import get_data, get_test # user func
from pprint import pprint
from datetime import datetime
import logging
import numpy as np
import pdb

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import normalize

from sklearn import cross_validation
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans

# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LassoCV
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.utils.extmath import density


models = [ ('ridge', RidgeClassifierCV(normalize = True)),
           ('lasso', LassoCV()),
         ]

parameters = {
         'ridge__C': (0.3, 1, 3, 10),
         'lasso__C': (1, 3),
}


def train_data(X, y, refit = False, test_size = 0.1):
    print("\n[ start training: {} ]".format(datetime.now()))
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size = test_size, random_state = 42)

#     scores = ["precision", "recall"]
    score = "recall"

#     for score in scores:
    for model in models:
        print("### Tuning parameters for {}\n".format(score))

        clf = GridSearchCV(Pipeline([model]), parameters, cv = 3, scoring = "recall", verbose = 1, n_jobs = -1)
        print(clf)
        clf.fit(xtrain, ytrain)

        print("\nBest parameters set found on development set:")
        print(clf.best_estimator_)

        print("\nGrid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))

        print("\nDetailed classification report:")
        y_true, y_pred = ytest, clf.predict(xtest)
        print(classification_report(y_true, y_pred))

    print("\n[ end training: {} ]".format(datetime.now()))

if __name__ == "__main__":
    print(__doc__)

    train = np.load("data/train.npy") 
    X = train[:,0:-1]
    y = train[:,-1]

    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))

    train_data(X, y, refit = False)
