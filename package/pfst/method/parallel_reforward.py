import numpy as np
from pfst.method.forward_backward_dropping import OneReforward
from pfst.method.trace_ratio import univariate
from pfst.utils.create_workers import divide_list
import multiprocessing
import itertools


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
    A_minus_R = np.setdiff1d(np.arange(X.shape[1]), list(R_after_forward_dropping))
    # print(f"Reset the selection pool to A \ R = {A_minus_R}")
    ##########################
    p = multiprocessing.Pool(processes = n_workers)
    blocks = divide_list(n_workers, A_minus_R)
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
        R_Reforward = pool.starmap(get_reforward, args_list)
    p.close()
    p.join()
    res = list(set(itertools.chain(*R_Reforward + [R_after_forward_dropping])))
    # print(f"After getting parallel Re-forward stage: {res}")
    return res

