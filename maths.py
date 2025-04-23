import numpy as np

def count(X):
    try:
        X = np.astype('float')
        X = X[~np.isnan(X)]
        return len(X)
    except :
        return len(X)