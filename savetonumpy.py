#!/usr/bin/env python2.7

import numpy as np
import pandas as pd
import pdb
import sys

training = pd.read_csv("data/training.csv", index_col = 0)
test = pd.read_csv("data/test.csv", index_col = 0)

print 0, len(training.columns)

#
# work on features
#
lab_train = training.pop("Label")
weight = training.pop("Weight")

print 1, len(training.columns)
print("Feature matrix columns:")
for col in training.columns:
    print u"  " + col

lab_train.iloc[np.where(lab_train == "s")] = 1
lab_train.iloc[np.where(lab_train == "b")] = 0

y_train = lab_train.values
weight = weight.values

X_train = training.values
X_test = test.values

print 2, X_train.shape

# pdb.set_trace()

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)

# print("X_train shape: {}".format(X_train.shape))
# print("Y_train shape: {}".format(y_train.shape))

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(y_train.shape))
print("Weight shape: {}".format(weight.shape))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-s":
        np.save("X_train.npy", X_train)
        np.save("Y_train.npy", y_train)
        print("Output written: X_train.npy, Y_train.npy")
        print("X_train shape: {}".format(X_train.shape))
        print("Y_train shape: {}".format(y_train.shape))

        np.save("X_test.npy", X_test)
        print("Output written: X_test.npy")
        print("X_test shape: {}".format(X_test.shape))

        np.save("weight.npy", weight)
        print("Output written: weight.npy")
        print("Weight shape: {}".format(weight.shape))
    else:
        print("---> No output, use -s flag to save")
