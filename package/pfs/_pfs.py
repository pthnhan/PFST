import numpy as np
from pfs.forward_backward_dropping import backward
from pfs.trace_ratio import univariate, mult_trace
from pfs.utils import divide_list
from pfs.parallel_forward_dropping import get_parallel_forward_dropping
from pfs.parallel_reforward import get_parallel_reforward
from pfs.parallel_R_argmax import get_parallel_get_argmax
import multiprocessing

R_backward = []

def get_R_backward(X, R_minus_f, y, ni, C):
    trace_univariate = np.array([univariate(X[:, i], y, ni, C) for i in np.arange(X.shape[1])])
    R = list(np.array([np.argmax(trace_univariate)]))
    return [R_minus_f[i] for i in R]

def parallel_forward_backward_algo(X, y, n_workers, alpha=0.05, beta=0.01):
    R_argmax = get_parallel_get_argmax(X, y, n_workers)
    R_forward_dropping = get_parallel_forward_dropping(X, y, n_workers, alpha)
    R_Reforward = get_parallel_reforward(X, y, n_workers, alpha)
    R = list(set(list(R_argmax) + list(R_forward_dropping) + list(R_Reforward)))
    blocks = divide_list(n_workers, R)
    p = multiprocessing.Pool(n_workers)
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    for block in blocks:
        R_minus_f = np.setdiff1d(R, block)
        X_block = X[:, R_minus_f]
        p.apply_async(get_R_backward,
                      args = (X_block, R_minus_f, y, ni, C),
                      callback = save_R_backward)
    p.close()
    p.join()
    return R_backward

def save_R_backward(R):
    R_backward.extend(R)


if __name__ == '__main__':
    from sklearn import datasets
    breastcc = datasets.load_breast_cancer()
    X, y = breastcc.data, breastcc.target.ravel()
    a = parallel_forward_backward_algo(X, y, n_workers = 2, alpha = 0.05, beta = 0.01)



