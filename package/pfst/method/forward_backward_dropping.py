# One forward dropping iteration:
# return: survived features, selected feature
import numpy as np
from pfst.method.trace_ratio import univariate, mult_trace

def OneForwardDropping(X, R, S, y, ni, g, alpha=0.05):
    if len(S) == 0:
        return 'drop block'

    if len(R) == 1:
        orig_trace = univariate(X[:, R], y, ni, g)
    else:
        orig_trace = mult_trace(X[:, R], y, ni, g)

    compute_t = lambda i: mult_trace(X[:, np.hstack((R, i))], y, ni, g)
    new_trace = np.array([compute_t(i) for i in S])
    trace_ratio = new_trace / orig_trace
    if (max(trace_ratio) - 1) > alpha:
        best = np.argmax(new_trace)
        return S[best], np.setdiff1d(S[np.where(trace_ratio > 1 + alpha)], S[best])
    else:
        return 'drop block'


# function that return the index of one column that should be added to the model
def OneReforward(X, R, S, y, ni, g, alpha):
    if len(S) == 0:
        return 'drop block'
    if len(R) == 1:
        orig_trace = univariate(X[:, R], y, ni, g)
    else:
        orig_trace = mult_trace(X[:, R], y, ni, g)
    compute_t = lambda i: mult_trace(X[:, np.hstack((R, i))], y, ni, g)
    new_trace = np.array([compute_t(i) for i in S])
    trace_diff = new_trace - orig_trace
    if (np.amax(new_trace) / orig_trace) > 1 + alpha:
        return S[np.argmax(new_trace)]
    else:
        return 'drop block'


def backward(X, R, y, ni, g, beta=0.01):
    flag = 0
    while flag != 'stop':
        orig_trace = mult_trace(X[:, R], y, ni, g)
        compute_t = lambda i: mult_trace(X[:, np.delete(R, i)], y, ni, g)
        new_trace = np.array([compute_t(i) for i in np.arange(len(R))])
        if max(new_trace) / orig_trace > 1 - beta:
            R = np.setdiff1d(R, np.argmin(new_trace))
        else:
            flag = 'stop'
    return R