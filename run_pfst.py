from pfst.utils.pfst_utils import log_results, timing, run, show_reults
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

@log_results
@timing
@run
def breast_cancer_data():
    from sklearn import datasets
    breastcc = datasets.load_breast_cancer()
    X, y = breastcc.data, breastcc.target.ravel()
    data = 'breast_cancer_data'
    n_workers = 5
    return X, y, data, n_workers

@log_results
@timing
@run
def parkinson_data():
    df = pd.read_csv("E:\study\ds_k30\paper\data\parkinson/pd_speech_features.csv")
    df = df[1:]
    df[df.columns] = df[df.columns].astype(float)
    X, y = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()
    y = y.astype(int)
    data = 'parkinson_data'
    n_workers = 10
    return X, y, data, n_workers


@log_results
@timing
@run
def gene_data():
    df_X = pd.read_csv("E:\study\ds_k30\paper\data\gene\data.csv")
    X = df_X[df_X.columns[1:]].to_numpy()
    df_y = pd.read_csv("E:\study\ds_k30\paper\data\gene\labels.csv")
    y = df_y[df_y.columns[-1]].to_numpy()

    X = X + 0.00001 * np.random.rand(X.shape[0], X.shape[1])
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    data = 'gene_data'
    n_workers = 10
    return X, y, data, n_workers


@log_results
@timing
@run
def micromass_data():
    label_data = pd.read_csv('E:\study\ds_k30\paper\data\micromass\mixed_spectra_metadata.csv', sep = ';')
    label = label_data[['Mixture_Label']]
    label = label.to_numpy()
    print(len(np.unique(label)))
    le2 = LabelEncoder()
    y = le2.fit_transform(label)

    df = pd.read_csv('E:\study\ds_k30\paper\data\micromass\mixed_spectra_matrix.csv', sep = ';', header = None)
    print(df.shape)
    print(df.head())
    X = df.to_numpy()
    var_vec = np.array([np.var(X[:, i]) for i in range(X.shape[1])])
    id = np.where(var_vec > 1e-5)
    X = X[:, id].reshape((len(X), -1))
    data = 'micromass_data'
    n_workers = 10
    return X, y, data, n_workers


@log_results
@timing
@run
def mutant_data():
    df = pd.read_table("E:\study\ds_k30\paper\data\mutant\K9.data")
    X = [np.array(list(df.columns)[0].split(",")[:-2]).astype(np.float)]
    y = [list(df.columns)[0].split(",")[-2]]
    for i in range(len(df)):
        if i != 1:
            val = df.loc[i].to_list()[0]
            val = val.replace("?", "0")
            X.append(np.array(val.split(",")[:-2]).astype(np.float))
            y.append(val.split(",")[-2])
    X = np.array(X)
    X = X + 0.00001 * np.random.rand(X.shape[0], X.shape[1])
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    data = 'mutant_data'
    n_workers = 10
    return X, y, data, n_workers

def main():
    ##### breast_cancer_data
    breast_cancer_data()

    # ##### Parkinson_data
    # parkinson_data()
    #
    # mutant_data()
    #
    # gene_data()
    #
    # micromass_data()

    ## show results
    show_reults()


if __name__ == '__main__':
    main()
