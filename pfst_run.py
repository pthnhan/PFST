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


if __name__ == '__main__':
    from pfs._pfs import get_pfs
    import time
    import numpy as np
    import pandas as pd
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    data = []
    runtime = []
    error = []
    num_R_selected = []
    #### breast_cancer_data
    from sklearn import datasets
    breastcc = datasets.load_breast_cancer()
    X, y = breastcc.data, breastcc.target.ravel()
    data.append('breast_cancer_data')
    t = time.time()
    R = get_pfs(X, y, n_workers=5, alpha=0.05, beta=0.01)
    res = [fold_k_err(X[:, R], y, k, LinearDiscriminantAnalysis()) for k in np.arange(5)]
    runtime.append(time.time() - t)
    error.append(np.mean(res))
    num_R_selected.append(len(R))

    #### Parkinson_data
    data.append("parkinson_data")
    df = pd.read_csv("/home/thanhnhan/Downloads/pd_speech_features.csv")
    df = df[1:]
    df[df.columns] = df[df.columns].astype(float)
    X, y = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()
    y = y.astype(int)
    t = time.time()
    R = get_pfs(X, y, n_workers=12, alpha=0.05, beta=0.01)
    res = [fold_k_err(X[:, R], y, k, LinearDiscriminantAnalysis()) for k in np.arange(5)]
    runtime.append(time.time() - t)
    error.append(np.mean(res))
    num_R_selected.append(len(R))

    df = pd.DataFrame()
    df['data'] = data
    df['error'] = error
    df['runtime'] = runtime
    df['num_R_selected'] = num_R_selected
    print(df)
    df.to_csv('psft.csv')
