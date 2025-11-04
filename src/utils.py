import os
import csv
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List, Optional, Iterable

# Reproducibilidad
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

# One-hot y métricas
def one_hot(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    y = y.astype(int).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh


def accuracy(y_pred_logits: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(y_pred_logits, axis=1)
    y_true = y_true.ravel()
    return float((y_pred == y_true).mean())


def confusion_matrix(
    y_pred_logits: np.ndarray, y_true: np.ndarray, num_classes: int
) -> np.ndarray:
    y_pred = np.argmax(y_pred_logits, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true.ravel(), y_pred):
        cm[int(yt), int(yp)] += 1
    return cm

# Particiones y mini-batches
def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Las particiones deben sumar 1.0"
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    te_idx = idx[n_train + n_val:]
    return (X[tr_idx], y[tr_idx]), (X[va_idx], y[va_idx]), (X[te_idx], y[te_idx])


def batch_iterator(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42
):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        yield X[idx], y[idx]

# Normalizaciones
def normalize_01(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mn, mx = X.min(), X.max()
    if mx - mn < 1e-12:
        return np.zeros_like(X)
    return (X - mn) / (mx - mn)


def standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd, mu, sd

# MNIST: carga local desde data/mnist/*.gz
def _load_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0


def _load_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)


def load_mnist(mnist_dir: str = "data/mnist"):
    req = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    missing = [f for f in req if not os.path.exists(os.path.join(mnist_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos MNIST en '{}': {}\n"
            "Copia manualmente los .gz en esa carpeta.".format(mnist_dir, missing)
        )

    X_train = _load_idx_images(os.path.join(mnist_dir, req[0]))
    y_train = _load_idx_labels(os.path.join(mnist_dir, req[1]))
    X_test  = _load_idx_images(os.path.join(mnist_dir, req[2]))
    y_test  = _load_idx_labels(os.path.join(mnist_dir, req[3]))
    return (X_train, y_train), (X_test, y_test)

# IRIS: carga local desde data/iris/iris.csv
_IRIS_CLASSES = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

def load_iris(iris_dir: str = "data/iris", filename: str = "iris.csv"):
    csv_path = os.path.join(iris_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró '{csv_path}'. Copia iris.csv en data/iris/."
        )

    X_list, y_list = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = [
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ]
                cls = row["class"]
                if cls not in _IRIS_CLASSES:
                    continue
                y_list.append(_IRIS_CLASSES[cls])
                X_list.append(x)
            except Exception:
                continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

# Visualización de curvas de entrenamiento
def plot_curves(history: Dict[str, List[float]]) -> None:
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy
    if "train_acc" in history:
        plt.figure()
        plt.plot(history["train_acc"], label="train_acc")
        if "val_acc" in history:
            plt.plot(history["val_acc"], label="val_acc")
        plt.xlabel("Épocas")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
