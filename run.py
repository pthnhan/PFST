import numpy as np
from pfst.run import run_pfst

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


def timing(f):
    import time
    def q(*args, **kwargs):
        t = time.time()
        f(*args, **kwargs)
        print(time.time() - t)

    return q


@timing
def main():
    import pandas as pd
    from sklearn import datasets
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    breastcc = datasets.load_breast_cancer()
    X, y = breastcc.data, breastcc.target.ravel()
    R = run_pfst(X, y, n_workers = 5, alpha = 0.05, beta = 0.01)
    print(R)
    # df = pd.read_csv("/home/thanhnhan/Downloads/pd_speech_features.csv")
    # df = df[1:]
    # df[df.columns] = df[df.columns].astype(float)
    # X, y = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()
    # print(X.shape, len(y))
    # R = get_pfs(X, y, n_workers = 12, alpha = 0.05, beta = 0.01)
    res = [fold_k_err(X[:, R], y, k, LinearDiscriminantAnalysis()) for k in np.arange(5)]
    print(np.mean(res))


if __name__ == '__main__':
    # main()
    # exit()
    import pandas as pd
    from sklearn import datasets
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import time
    breastcc = datasets.load_breast_cancer()
    X, y = breastcc.data, breastcc.target.ravel()
    t = time.time()
    R = run_pfst(X, y, n_workers = 5, alpha = 0.05, beta = 0.01)
    print(R)
    res = [fold_k_err(X[:, R], y, k, LinearDiscriminantAnalysis()) for k in np.arange(5)]
    print(np.mean(res))
    print(time.time() - t)
