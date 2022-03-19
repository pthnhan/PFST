import numpy as np
from pfst.method.forward_backward_dropping import OneForwardDropping
from pfst.method.trace_ratio import univariate
from pfst.utils.create_workers import divide_list
import multiprocessing
import itertools


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
    p = multiprocessing.Pool(processes = n_workers)
    blocks = divide_list(n_workers, list(range(X.shape[1])))
    args_list = []
    for block in blocks:
        X_block = X[:, block]
        n_features = X_block.shape[1]
        C = len(np.unique(y))
        ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
        trace_univariate = np.array([univariate(X_block[:, i], y, ni, C) for i in np.arange(X_block.shape[1])])
        R = np.array([np.argmax(trace_univariate)])
        S = np.setdiff1d(np.arange(n_features), R)
        args_list.append((X_block, block, R, S, y, ni, C, alpha))
    with multiprocessing.Pool(processes = n_workers) as pool:
        R_forward_dropping = pool.starmap(get_one_forward_dropping, args_list)
    res = list(set(itertools.chain(*R_forward_dropping + [R_after_argmax])))
    # print(f"After getting parallel forward with forward-dropping stage: R = {res}")
    return res
