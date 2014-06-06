#!/usr/bin/env python2.7

import numpy as np
import pandas as pd
import pdb
import sys

training = pd.read_csv("training.csv", index_col = 0)
test = pd.read_csv("test.csv", index_col = 0)
# testdf = pd.read_csv("test.csv")

# print("sample size: {}".format(len(traindf)))
# sampleindex = traindf.pop(u"Unnamed: 0")

# cpubusy = traindf.pop("cpu_01_busy")
# y = np.mat(cpubusy.values).T

#
# work on features
#
lab_train = training.pop("Label")

pdb.set_trace()

lab_train.iloc[np.where(lab_train == "s")] = 1
lab_train.iloc[np.where(lab_train == "b")] = 0

y_train = np.mat(lab_train).T
# m = y.shape[0]

# turn time into integer 
# dttime = pd.to_datetime(traindf.pop("sample_time"))
# inttime = dttime.astype(np.int64)

# del traindf["syst_page_read_ipo_rate"]
# del traindf["page_global_valid_fault_rate"]

# X = np.mat(np.c_[np.ones(m), traindf.values, m_id_dummy, inttime])
X_train = np.mat(training.values)
X_test = np.mat(test.values)
# X = np.mat(np.c_[training.values])

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-s":
        np.save("X_train.npy", X_train)
        np.save("Y_train.npy", y_train)
        print("Output written:\nX_train.npy\nY_train.npy")
        print("X_train shape: {}".format(X_train.shape))
        print("Y_train shape: {}".format(y_train.shape))

        np.save("X_test.npy", X_test)
        print("Output written:\nX_test.npy")
        print("X_test shape: {}".format(X_test.shape))
    else:
        print("no output")
