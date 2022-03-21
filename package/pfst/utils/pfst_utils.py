import numpy as np
from pfst.run import run_pfst
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = []
runtimes = []
error = []
num_R_selected = []

def fold_k_err(X, y, k, model):
    np.random.seed(1)
    fold = np.random.choice(np.arange(5), len(y))
    Xtrain, ytrain = X[fold != k, :], y[fold != k]
    Xtest, ytest = X[fold == k, :], y[fold == k]
    # print(Xtrain, ytrain, Xtest, ytest)
    clf = model
    clf.fit(Xtrain, ytrain)
    test_pred = clf.predict(Xtest)
    err = np.mean(test_pred != ytest)
    return err


def log_results(f):
    def inner():
        res, runtime = f()
        data.append(res[0])
        num_R_selected.append(res[1])
        error.append(res[2])
        runtimes.append(runtime)

    return inner


def timing(f):
    import time
    def inner():
        t = time.time()
        return f(), time.time() - t

    return inner


def run(f):
    def inner():
        X, y, data, n_workers = f()
        R = run_pfst(X, y, n_workers = n_workers, alpha = 0.05, beta = 0.01)
        err = [fold_k_err(X[:, R], y, k, LinearDiscriminantAnalysis()) for k in np.arange(5)]
        return data, len(R), np.mean(err)

    return inner


def show_reults():
    df = pd.DataFrame()
    df['data'] = data
    df['error'] = error
    df['runtime'] = runtimes
    df['num_R_selected'] = num_R_selected
    print(df)
    df.to_csv('psft.csv')