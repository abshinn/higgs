#!/usr/bin/env python2.7

import sklearn as sklearn
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import hinge_loss
from time import time
from sklearn.pipeline import Pipeline
import pprint as pprint


def create_scores(model,X,y):
	scores=cross_validation.cross_val_score(model,X,y,cv=5)
	return scores

def LR_func(X,y):
    clf = LR()
    return clf.fit(X,y)

def SVC_func(kernel,X,y):
	clf = SVC(kernel=kernel,class_weight=auto)
	return clf.fit(X,y)

def GridSearch_func(model,X,y,params):
	clf = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
	clf.fit(X,y)

# pipeline = Pipeline([
# 	#('clf_LR', LR()),
# 	('clf_SVC_poly', SVC(kernel='poly'))
# ])

# #parameters = {'C':[.001,.1,1,1000],'degree':[1,2,3]}
# parameters = {'degree':[1,2,3]}


if __name__ == "__main__":
	
	X = np.load("data/X_train.npy")
	print X.shape
	# X = X[:,2:5]
	y = np.load("data/y_train.npy")
	y = np.squeeze(np.asarray(y))
	print y.shape

	X = scale(X)

	# kernels = ['linear', 'poly', 'rbf']

	# clf_SVC_poly = SVC(kernel='poly')

	# parameters = {'C':[.001,.1,1,1000],'degree':[1,2,3]}
	parameters = {'C':[.001],'degree':[1]}
	
	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 42)

	# clf_SVC_poly = SVC(kernel='poly')
	clf = LR(C = 10)
	# clf.fit(train_X,train_y)
	scores=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
	print scores

	#grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='roc_auc')
	# grid_search = GridSearchCV(clf, parameters, cv=5, scoring='roc_auc')
	# grid_search.fit(X,y)

	# # print clf_grid.grid_scores_
	# # print clf_grid.best_params_
	# print("Performing grid search...")
	# print("pipeline:", [name for name, _ in pipeline.steps])
	# print("parameters:")
	# pprint(parameters)
	# t0 = time()
	# grid_search.fit(xtrain, ytrain)
	# print("done in %0.3fs" % (time() - t0))
	# print()

	# print("Best score: %0.3f" % grid_search.best_score_)
	# print("Best parameters set:")
	# best_parameters = grid_search.best_estimator_.get_params()
	# for param_name in sorted(parameters.keys()):
	# 	print("\t%s: %r" % (param_name, best_parameters[param_name]))

