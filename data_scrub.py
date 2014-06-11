import numpy as np
from sklearn.preprocessing import scale

def get_data(exp_classifier = True, scaleX = True):
    # load data
    X = np.load("data/X_train.npy")
    y = np.load("data/y_train.npy")
    # weight = np.load("data/weight.npy")

    print("\n### loaded X, shape: {}".format(X.shape))
     
    # print("\n### without NA ###")
    # na_ind = np.where(X == -999.)
    # X_na = X[na_ind[0]]
    # y_na = y[na_ind[0]]
    # X = np.delete(X, na_ind[0], axis = 0)
    # y = np.delete(y, na_ind[0], axis = 0)

    if exp_classifier:
        print("### na classifier: 1./exp(sum(nans), axis = 1)")
        # na_class = np.ones(y.shape)
        # na_class[na_ind[0]] = 0
        # X = np.c_[X, na_class] 
        na_class = 1./np.exp((X == -999.).sum(axis = 1))
        X = np.c_[X, na_class]

    if scaleX:
        print("### scale X")
        X = scale(X)

    print("### returning X, y of shapes: {}, {}".format(X.shape, y.shape))

    return X, y


def get_test(exp_classifier = True, scaleX = True):
    # load test data
    Xtest = np.load("data/X_test.npy")
    ID = Xtest[:,0]
    # weight = np.load("data/weight.npy")

    print("\n### loaded X_test, shape: {}".format(Xtest.shape))
     
    # print("\n### without NA ###")
    # na_ind = np.where(X == -999.)
    # X_na = X[na_ind[0]]
    # y_na = y[na_ind[0]]
    # X = np.delete(X, na_ind[0], axis = 0)
    # y = np.delete(y, na_ind[0], axis = 0)

    if exp_classifier:
        print("### na classifier: 1./exp(sum(nans), axis = 1)")
        # na_class = np.ones(y.shape)
        # na_class[na_ind[0]] = 0
        # X = np.c_[X, na_class] 
        na_class = 1./np.exp((Xtest == -999.).sum(axis = 1))
        Xtest = np.c_[Xtest, na_class]

    if scaleX:
        print("### scale X")
        Xtest = scale(Xtest)

    print("### returning X_test of shape: {}".format(Xtest.shape))

    return Xtest, ID
