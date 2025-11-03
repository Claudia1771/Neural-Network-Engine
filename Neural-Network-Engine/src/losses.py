import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / m

def cross_entropy_grad(y_true, y_pred):
    m = y_true.shape[0]
    return (y_pred - y_true) / m

def mse(y_true, y_pred):
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def mse_grad(y_true, y_pred):
    m = y_true.shape[0]
    return (y_pred - y_true) / m
