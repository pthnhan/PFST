import numpy as np
from pfs.trace_ratio import univariate
from pfs.utils import divide_list
import multiprocessing

R_argmax = []


def get_argmax(X, block, y, ni, C):
    trace_univariate = np.array([univariate(X[:, i], y, ni, C) for i in np.arange(X.shape[1])])
    R = list(np.array([np.argmax(trace_univariate)]))
    return [block[i] for i in R]


def get_parallel_get_argmax(X, y, n_workers):
    p = multiprocessing.Pool(n_workers)
    blocks = divide_list(n_workers, list(range(X.shape[1])))
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    for block in blocks:
        X_block = X[:, block]
        p.apply_async(get_argmax,
                      args = (X_block, block, y, ni, C),
                      callback = save_R_argmax)
    p.close()
    p.join()
    return R_argmax


def save_R_argmax(R):
    R_argmax.extend(R)
