import numpy as np
from numpy.linalg import inv

# return trace ratio for univariate case
def univariate(x,y,ni, C):
    x = x.flatten()
    xbar = np.mean(x)
    num_i = lambda i: sum(y==i)*(np.mean(x[y==i])-xbar)**2
    numerator = sum(np.array([num_i(i) for i in np.arange(C)]))
    den_i = lambda i: (ni[i]-1)*np.var(x[y==i])
    denominator = sum(np.array([den_i(i) for i in np.arange(C)]))
    return numerator/denominator #np.divide(class_mean_diff, denominator)

# compute trace ratio
def mult_trace(Xmat,y,ni,g):
    xbar = np.mean(Xmat, axis = 0)
    Sw = 0
    Sb = 0
    for cl in np.arange(g):
        u = np.mean(Xmat[y==cl,:], axis=0)-xbar
        Sb += ni[cl]*np.outer(u,u)
        Sw += (ni[cl]-1)*np.cov(Xmat[y==cl,:].T)
    return np.trace(np.matmul(inv(Sw),Sb))