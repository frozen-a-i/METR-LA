import numpy as np


def mae(a, b):
    return np.mean(np.abs(a - b))

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mape(a, b, eps=1.0):
    m = np.abs(a) > eps
    return np.mean(np.abs((a[m] - b[m]) / a[m])) * 100
