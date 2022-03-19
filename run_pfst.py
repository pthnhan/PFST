from pfst.utils.pfst_utils import log_results, timing, run, show_reults
import pandas as pd

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
    df = pd.read_csv("/home/thanhnhan/Downloads/pd_speech_features.csv")
    df = df[1:]
    df[df.columns] = df[df.columns].astype(float)
    X, y = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()
    y = y.astype(int)
    data = 'parkinson_data'
    n_workers = 12
    return X, y, data, n_workers

def main():
    ##### breast_cancer_data
    breast_cancer_data()

    ##### Parkinson_data
    parkinson_data()

    ## show results
    show_reults()


if __name__ == '__main__':
    main()