import numpy as np
import os

def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]; y = y[idx]
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def get_batches(X, y, batch_size, shuffle=True, seed=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bidx = idx[start:end]
        yield X[bidx], y[bidx]

def load_iris():
    try:
        from sklearn.datasets import load_iris as _load_iris
        data = _load_iris()
        X = data["data"].astype(np.float32)
        y = data["target"].astype(int)
        return X, y
    except ImportError:
        raise ImportError("Instala scikit-learn o coloca iris.csv en /data")

def load_mnist_from_npz(path="data/mnist.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError("Falta data/mnist.npz")
    f = np.load(path)
    X_train, y_train = f["x_train"], f["y_train"]
    X_test, y_test = f["x_test"], f["y_test"]
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.0
    return (X_train, y_train), (X_test, y_test)
