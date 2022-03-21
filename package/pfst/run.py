import numpy as np
from pfst.method.trace_ratio import mult_trace
from pfst.method.parallel_forward_dropping import get_parallel_forward_dropping
from pfst.method.parallel_reforward import get_parallel_reforward
from pfst.method.parallel_R_argmax import get_parallel_get_argmax
from pfst.method.parallel_backward_argmax import get_parallel_backward_argmax


def run_pfst(X, y, n_workers=5, alpha=0.05, beta=0.01, gamma=0.05):
    print("PFST is starting...")
    R_after_argmax = get_parallel_get_argmax(X, y, n_workers)
    print("Stage 1: Forward with forward-dropping stage")
    R_after_forward_dropping = get_parallel_forward_dropping(X, y, n_workers, R_after_argmax, alpha)
    ##################
    print("Stage 2: Re-forward stage!")
    R_after_Reforward = get_parallel_reforward(X, y, n_workers, R_after_forward_dropping, alpha)
    print("Stage 3: Backward stage")
    set_argmax_R_after_Reforward = get_parallel_backward_argmax(X, y, R_after_Reforward, n_workers)
    C = len(np.unique(y))
    ni = np.array([np.sum(y == cl) for cl in np.arange(C)])
    compute_t = lambda i: mult_trace(X[:, np.delete(R_after_Reforward, i)], y, ni, C)
    set_trace = np.array([compute_t(i) for i in range(len(set_argmax_R_after_Reforward))])
    print(f"Backward... To get R_selected!")
    f_r = np.argmax(set_trace)
    t_R = mult_trace(X[:, R_after_Reforward], y, ni, C)
    t_R_minus_f_r = compute_t(f_r)
    if t_R - t_R_minus_f_r < beta:
        R_selected = list(np.delete(R_after_Reforward, f_r))
    else:
        R_selected = list(R_after_Reforward)
    print(f"R_selected: {R_selected}")
    return R_selected
