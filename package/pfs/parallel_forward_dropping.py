import numpy as np
from pfs.forward_backward_dropping import OneForwardDropping
from pfs.trace_ratio import univariate
from pfs.utils import divide_list
import multiprocessing

R_forward_dropping = []


def get_one_forward_dropping(X, block, R, S, y, ni, C, alpha = 0.05):
    result = 0
    while result != 'drop block':
        result = OneForwardDropping(X, R, S, y, ni, C, alpha = alpha)
        if result != 'drop block':
            # get the element
            R_get, S = result[0], result[1]

            # form the result R.
            R = np.hstack((R, R_get))
    print(f"{block} -> {[block[i] for i in R]}")
    return [block[i] for i in R]


def get_parallel_forward_dropping(X, y, n_workers, R_after_argmax, alpha=0.05, gamma=0.05):
    print(f"Start getting parallel forward dropping with {n_workers} workers!")
    p = multiprocessing.Pool(n_workers)
    blocks = divide_list(n_workers, list(range(X.shape[1])))
    for block in blocks:
        X_block = X[:, block]
        n_features = X_block.shape[1]
        C = len(np.unique(y))
        ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
        trace_univariate = np.array([univariate(X_block[:, i], y, ni, C) for i in np.arange(X_block.shape[1])])
        R = np.array([np.argmax(trace_univariate)])
        S = np.setdiff1d(np.arange(n_features), R)
        p.apply_async(get_one_forward_dropping,
                      args = (X_block, block, R, S, y, ni, C, alpha),
                      callback = save_R_forward_dropping)
    p.close()
    p.join()
    print(f"After getting parallel forward with forward-dropping stage: R = {list(set(list(R_after_argmax) + list(R_forward_dropping)))}")
    return list(set(list(R_after_argmax) + list(R_forward_dropping)))


def save_R_forward_dropping(R):
    R_forward_dropping.extend(R)
