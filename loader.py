import numpy as np
import pandas as pd


def load_data(data_path):
    images = None
    labels = None
    df = pd.read_csv(data_path)
    images = np.asanyarray(df.drop('label', axis=1))
    labels = np.asanyarray(df[['label']])
    labels = np.reshape(labels, (len(labels)))
    return images, labels
