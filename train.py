#!/usr/bin/env python2.7
"""train higgs boson data on one model, multiple params..."""

from data_scrub import get_data, get_test # user func
from pprint import pprint
from datetime import datetime
import logging

from sklearn.metrics import roc_auc_score
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import NearestCentroid
# from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import numpy as np

# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
#         ('rdge', RidgeClassifier()),
#         ('kNN', KNeighborsClassifier(n_neighbors = 55)),
#         ('SVC', LinearSVC()),
          ('SVC', SVC()),
#         ( 'clf', LogisticRegression(penalty = 'l1')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
#        'rdge__alpha': (0.01, 0.1, 1, 10),
#        'kNN__n_neighbors': (55,60,65),
#        'clf__C': (1, 30),
#        'clf__penalty': ('l1'),
#        'clf__n_iter': (10, 50, 80),
         'SVC__C': (3, 4),
#        'SVC__loss': ('l1', 'l2'),
}


def train_data(X, y, refit = False):
    print("\n[ start training: {} ]".format(datetime.now()))
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size = 0.1, random_state = 42)

    # multiprocessing requires the fork to happen in a __main__ protected block
    grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = "roc_auc", refit = refit)

    print("\nPerforming grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = datetime.now()
    grid_search.fit(xtrain, ytrain)
    print("done in {}\n".format(datetime.now() - t0))

    print("all scores:")
    for item in grid_search.grid_scores_:
        print item

    print("\nBest score: %0.3f" % grid_search.best_score_)
    print("\nBest parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("\n[ end training: {} ]".format(datetime.now()))

    if refit:
        Xtest, ID = get_test()

        y_vals = grid_search.predict(Xtest)
        y_out = np.array(range(1, len(y_vals)+1)).astype(str)
        y_out[np.where(y_vals == 1)] = 's'
        y_out[np.where(y_vals == 0)] = 'b'

        rank = np.arange(1,len(y_vals)+1,1)

        ID = ID.astype(int).astype(str)
        output = np.c_[ID, rank, y_out]

        output.astype("|S5")
        np.savetxt("submission.csv", output, delimiter=",", fmt="%s")


if __name__ == "__main__":
    print(__doc__)

    X, y = get_data()

    train_data(X, y, refit = True)
