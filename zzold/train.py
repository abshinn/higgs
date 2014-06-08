#!/usr/bin/env python2.7

import sklearn as sklearn
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import hinge_loss
from time import time
from sklearn.pipeline import Pipeline
import pprint as pprint
import pdb

X = np.load("data/X_train.npy")
# X_realtest = np.load("data/X_test.npy")
#print X.shape
# X = X[:,0:30]
y = np.load("data/y_train.npy")
# y = np.squeeze(np.asarray(y))


print X.shape

# pdb.set_trace()

X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# clf_SVC_poly = SVC(kernel='poly',C=.001, degree=1)
clf = LR(C = 10)
# clf = Ridge()
# clf = PAC(C = 10)

clf.fit(X_train,y_train)
scores = cross_val_score(clf,X_train,y_train,cv=5,scoring="accuracy")

print scores


if 0:
    y_vals = clf.predict(X_realtest)
    y_out = np.zeros(len(y_vals))
    y_out[np.where(y_vals == 1)] = 's'
    y_out[np.where(y_vals == 0)] = 'b'
    rank = np.arange(1,len(y_vals)+1,1)
    ID = X_realtest[:,0]
    output = np.c_[ID, rank, y_vals]
    np.savetxt("submission.csv", output, delimiter=",")
