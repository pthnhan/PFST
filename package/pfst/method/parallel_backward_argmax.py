import numpy as np
from pfst.method.trace_ratio import mult_trace
from pfst.utils.create_workers import divide_list
import multiprocessing


def get_R_backward_argmax(X, block, R, y, ni, C):
    compute_t = lambda i: mult_trace(X[:, np.delete(R, i)], y, ni, C)
    block_trace = np.array([compute_t(R.index(block[i])) for i in range(len(block))])
    print(f"{block} -> {[block[np.argmax(block_trace)]]}")
    return block[np.argmax(block_trace)]


def get_parallel_backward_argmax(X, y, R_after_Reforward, n_workers):
    print(f"Start getting parallel set argmax of R_after_Reforward with {n_workers} workers!")
    blocks = divide_list(n_workers, R_after_Reforward)
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    with multiprocessing.Pool(processes = n_workers) as pool:
        args_list = [(X, block, R_after_Reforward, y, ni, C) for block in blocks]
        set_argmax_R_after_Reforward = pool.starmap(get_R_backward_argmax, args_list)
    # print(f"After getting parallel R backward argmax: R = {R_backward_argmax}")
    return set_argmax_R_after_Reforward