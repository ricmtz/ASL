import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib

TRAIN_DATA_PATH = 'sign-language-mnist/sign_mnist_train.csv'
TEST_DATA_PATH = 'sign-language-mnist/sign_mnist_test.csv'


def load_data(data_path):
    df = pd.read_csv(data_path)
    x = np.asanyarray(df.drop('label', axis=1))
    y = np.asanyarray(df[['label']])
    y = np.reshape(y, (len(y)))
    return x, y


def train_pca():
    n_components = 300
    pca = PCA(n_components=n_components)
    x, y = load_data(TRAIN_DATA_PATH)
    pca.fit(x)
    joblib.dump(pca, 'PCA.joblib')


def get_pca():
    pca = joblib.load('PCA.joblib')
    return pca


def reduce_dimen(data):
    pca = get_pca()
    return pca.transform(data)
