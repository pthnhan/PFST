import numpy as np
from pfs.forward_backward_dropping import OneReforward
from pfs.trace_ratio import univariate, mult_trace
from pfs.utils import divide_list
import multiprocessing
from pfs.parallel_forward_dropping import get_parallel_forward_dropping

R_Reforward = []


def get_reforward(X, block, R, S, y, ni, C, alpha = 0.05):
    result = 0
    while result != 'drop block':
        result = OneReforward(X, R, S, y, ni, C, alpha = alpha)
        if result != 'drop block':
            R = np.hstack((R, result))
            S = np.setdiff1d(S, result)
    print(f"{block} -> {[block[i] for i in R]}")
    return [block[i] for i in R]


def get_parallel_reforward(X, y, n_workers, R_after_forward_dropping, alpha = 0.05, gamma = 0.05):
    print(f"Start getting parallel reforward with {n_workers} workers!")
    A_minus_R = np.setdiff1d(np.arange(X.shape[1]), R_after_forward_dropping)
    print(f"Reset the selection pool to A \ R = {A_minus_R}")
    ##########################
    p = multiprocessing.Pool(n_workers)
    blocks = divide_list(n_workers, A_minus_R)
    for block in blocks:
        X_block = X[:, block]
        n_features = X_block.shape[1]
        C = len(np.unique(y))
        ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
        trace_univariate = np.array([univariate(X_block[:, i], y, ni, C) for i in np.arange(X_block.shape[1])])
        R = np.array([np.argmax(trace_univariate)])
        S = np.setdiff1d(np.arange(n_features), R)
        p.apply_async(get_reforward,
                      args = (X_block, block, R, S, y, ni, C, alpha),
                      callback = save_R_reforward)
    p.close()
    p.join()
    print(f"After getting parallel Re-forward stage: {list(set(list(R_after_forward_dropping) + list(R_Reforward)))}")
    return list(set(list(R_after_forward_dropping) + list(R_Reforward)))


def save_R_reforward(R):
    R_Reforward.extend(R)
