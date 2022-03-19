import numpy as np
from pfst.method.trace_ratio import univariate
from pfst.method.forward_backward_dropping import OneForwardDropping, OneReforward, backward


def feature_selection(X, y, k, alpha, beta):
    # find the most relevant feature to be included in the model
    n_features = X.shape[1]
    result = 0
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])

    # Forward steps
    trace_univariate = np.array([univariate(X[:, i], y, ni, C) for i in np.arange(X.shape[1])])
    R = np.array([np.argmax(trace_univariate)])
    S = np.setdiff1d(np.arange(n_features), R)
    while result != 'drop block':
        result = OneForwardDropping(X, R, S, y, ni, C, alpha = alpha)
        if result != 'drop block':
            # get the element
            R_get, S = result[0], result[1]

            # form the result R.
            R = np.hstack((R, R_get))
            print(R)

    # print("---------------1-----------------")
    # reforward steps
    result = 0
    S = np.setdiff1d(np.arange(n_features), R)
    while result != 'drop block':
        result = OneReforward(X, R, S, y, ni, C, alpha = alpha)
        if result != 'drop block':
            R = np.hstack((R, result))
            S = np.setdiff1d(S, result)
            print("k={}, R={}".format(k, R))

    # print("---------------2-----------------")
    R = backward(X, R, y, ni, C, beta = beta)
    print("k={}, keep_R:".format(k, R))
    return R
