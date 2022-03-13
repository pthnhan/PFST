import numpy as np
from pfs.forward_backward_dropping import backward
from pfs.trace_ratio import univariate, mult_trace
from pfs.utils import divide_list
from pfs.parallel_forward_dropping import get_parallel_forward_dropping
from pfs.parallel_reforward import get_parallel_reforward
from pfs.parallel_R_argmax import get_parallel_get_argmax
import multiprocessing

R_backward_argmax = []


def get_R_backward_argmax(X, block, R, y, ni, C):
    compute_t = lambda i: mult_trace(X[:, np.delete(R, i)], y, ni, C)
    block_trace = np.array([compute_t(R.index(block[i])) for i in range(len(block))])
    print(f"{block} -> {[block[np.argmax(block_trace)]]}")
    return [block[np.argmax(block_trace)]]


def get_parallel_backward_argmax(X, y, n_workers, alpha=0.05):
    R_after_argmax = get_parallel_get_argmax(X, y, n_workers)
    print("Stage 1: Forward with forward-dropping stage")
    R_after_forward_dropping = get_parallel_forward_dropping(X, y, n_workers, R_after_argmax, alpha)
    ##################
    print("Stage 2: Re-forward stage!")
    R_after_Reforward = get_parallel_reforward(X, y, n_workers, R_after_forward_dropping, alpha)
    print("Stage 3: Backward stage")
    print(f"Start getting parallel R backward argmax with {n_workers} workers!")
    blocks = divide_list(n_workers, R_after_Reforward)
    p = multiprocessing.Pool(n_workers)
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    for block in blocks:
        p.apply_async(get_R_backward_argmax,
                      args = (X, block, R_after_Reforward, y, ni, C),
                      callback = save_R_backward_argmax)
    p.close()
    p.join()
    print(f"After getting parallel R backward argmax: R = {list(set(R_backward_argmax))}")
    return list(set(R_backward_argmax))


def save_R_backward_argmax(R):
    R_backward_argmax.extend(R)


def get_pfs(X, y, n_workers=5, alpha=0.05, beta=0.01, gamma=0.05):
    print("PFST is starting...")
    R = get_parallel_backward_argmax(X, y, n_workers, alpha)
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    print(f"Backward... To get R_selected!")
    R_selected = backward(X, R, y, ni, C, beta)
    print(f"R_selected: {R_selected}")
    return R_selected