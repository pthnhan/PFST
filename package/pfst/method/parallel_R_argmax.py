import numpy as np
from pfst.method.trace_ratio import univariate
from pfst.utils.create_workers import divide_list
import multiprocessing


def get_argmax(X, block, y, ni, C):
    trace_univariate = np.array([univariate(X[:, i], y, ni, C) for i in np.arange(X.shape[1])])
    print(f"{block} -> {[block[np.argmax(trace_univariate)]]}")
    return block[np.argmax(trace_univariate)]


def get_parallel_get_argmax(X, y, n_workers):
    print(f"Start getting parallel argmax with {n_workers} workers!")
    # p = multiprocessing.Pool(processes = n_workers)
    blocks = divide_list(n_workers, list(range(X.shape[1])))
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    args_list = []
    for block in blocks:
        X_block = X[:, block]
        args_list.append((X_block, block, y, ni, C))
    with multiprocessing.Pool(processes = n_workers) as pool:
        R_argmax = pool.starmap(get_argmax, args_list)
    # print(f"After getting parallel argmax: R = {R_argmax}")
    return R_argmax

