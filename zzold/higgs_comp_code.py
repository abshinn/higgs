
# coding: utf-8

# In[1]:

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


# In[2]:

X = np.load("X_train.npy")
X_realtest = np.load("X_test.npy")
#print X.shape
#X = X[:,0:30]
y = np.load("y_train.npy")
y = np.squeeze(np.asarray(y))


# In[3]:

X = scale(X)


# In[4]:

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)


# In[5]:

#clf_SVC_poly = SVC(kernel='poly',C=.001, degree=1)
clf = LR()
clf.fit(train_X,train_y)
# In[ ]:

#clf_SVC_poly.fit(train_X,train_y)


# In[ ]:

#y_vals = clf_SVC_poly.predict(X_realtest)
y_vals = clf.predict(X_realtest)
y_out = np.array(range(1, len(y_vals)+1)).astype(str)
y_out[np.where(y_vals == 1)] = 's'
y_out[np.where(y_vals == 0)] = 'b'


# In[ ]:

print y_out[0:10]

rank = np.arange(1,len(y_vals)+1,1)

print rank[0:10]

ID = X_realtest[:,0].astype(int).astype(str)
output = np.c_[ID, rank, y_out]

print rank
print ID

# In[ ]:

print output[0:10,:]

output.astype("|S5")
print output.dtype
np.savetxt("submission.csv", output, delimiter=",", fmt="%s")

