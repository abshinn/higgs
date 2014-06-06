#!/usr/bin/env python2.7

import numpy as np
import pandas as pd
import pdb
import sys

traindf = pd.read_csv("training.csv")
# testdf = pd.read_csv("test.csv")

pdb.set_trace()

# print("sample size: {}".format(len(traindf)))
# sampleindex = traindf.pop(u"Unnamed: 0")

cpubusy = traindf.pop("cpu_01_busy")
y = np.mat(cpubusy.values).T

#
# work on features
#
m = y.shape[0]
label = traindf.pop("Label")

label[np.where(label == "s")] = 1
label[np.where(label == "b")] = 0


# binarization of server id
# m_id_dummy = pd.get_dummies(m_id).values[:,1:]

# turn time into integer 
dttime = pd.to_datetime(traindf.pop("sample_time"))
inttime = dttime.astype(np.int64)

# del traindf["syst_page_read_ipo_rate"]
# del traindf["page_global_valid_fault_rate"]

# X = np.mat(np.c_[np.ones(m), traindf.values, m_id_dummy, inttime])
X = np.mat(np.c_[traindf.values, m_id_dummy, inttime])

X = X.astype(np.float64)
y = y.astype(np.float64)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-s":
        np.save("X.npy", X)
        np.save("Y.npy", y)
        print("Output written:\nX.npy\nY.npy")
    else:
        print("no output")
