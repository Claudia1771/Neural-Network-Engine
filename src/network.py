import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

from .layers import Layer, Dense, Dropout
from .losses import Loss
from .optimizers import Optimizer
from . import utils

class NeuralNetwork:
    """Red neuronal secuencial basada en lista de capas."""

    def __init__(self, layers: List[Layer]):
        assert len(layers) > 0, "Debe haber al menos una capa"
        self.layers = layers
        self._last_inputs = None
        self._last_logits = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagación hacia adelante."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        self._last_inputs = X
        self._last_logits = out
        return out

    def backward(self, grad: np.ndarray) -> None:
        """Propagación hacia atrás."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self) -> Dict[str, np.ndarray]:
        """Devuelve parámetros con nombres estables por capa."""
        p: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params().items():
                p[f"{k}_{i}"] = v
        return p

    def grads(self) -> Dict[str, np.ndarray]:
        """Devuelve gradientes con nombres estables por capa."""
        g: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads().items():
                g[f"{k}_{i}"] = v
        return g

    def zero_grad(self):
        pass

    def save(self, path: str):
        """Guarda los parámetros en un .npz."""
        np.savez(path, **self.params())

    def load(self, path: str):
        """Carga los parámetros desde un .npz."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for k in layer.params().keys():
                layer.params()[k][:] = data[f"{k}_{i}"]

# ---- Schedulers ----

def step_decay(epoch: int, base_lr: float, drop: float = 0.5, every: int = 10) -> float:
    """Reduce el learning rate cada cierto número de épocas."""
    return base_lr * (drop ** (epoch // every))

def cosine_annealing(epoch: int, base_lr: float, max_epochs: int) -> float:
    """Cosine annealing para el learning rate."""
    return 0.5 * base_lr * (1 + np.cos(np.pi * epoch / max_epochs))

class Trainer:
    """Clase de entrenamiento con validación y early stopping."""

    def __init__(
        self,
        network: NeuralNetwork,
        optimizer: Optimizer,
        loss_fn: Loss,
        weight_decay: float = 0.0,
        early_stopping: bool = True,
        patience: int = 5,
        lr_scheduler: Optional[Callable[[int, float], float]] = None,
        base_lr: Optional[float] = None,
        verbose: bool = True,
        seed: int = 42
    ):
        self.net = network
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.base_lr = base_lr
        self.verbose = verbose
        self.seed = seed

    def _evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
        """Evalúa loss y accuracy sin dropout."""
        losses, accs = [], []
        for xb, yb in utils.batch_iterator(X, y, batch_size=batch_size, shuffle=False):
            logits = self.net.forward(xb, training=False)
            loss = self.loss_fn.forward(logits, yb)
            losses.append(loss)
            y_true = yb if yb.ndim == 1 else np.argmax(yb, axis=1)
            accs.append(utils.accuracy(logits, y_true))
        return float(np.mean(losses)), float(np.mean(accs))

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20, batch_size: int = 64
    ):
        history = {"train_loss": [], "train_acc": []}
        if X_val is not None:
            history.update({"val_loss": [], "val_acc": []})

        best_val = float("inf")
        patience_left = self.patience
        base_lr = self.base_lr

        for epoch in range(1, epochs + 1):
            if self.lr_scheduler is not None and base_lr is not None:
                if hasattr(self.opt, "lr"):
                    if self.lr_scheduler.__name__ == "cosine_annealing":
                        self.opt.lr = self.lr_scheduler(epoch - 1, base_lr, max_epochs=epochs)
                    else:
                        self.opt.lr = self.lr_scheduler(epoch - 1, base_lr)

            train_losses, train_accs = [], []

            for xb, yb in utils.batch_iterator(
                X_train, y_train, batch_size=batch_size, shuffle=True, seed=self.seed + epoch
            ):
                logits = self.net.forward(xb, training=True)
                loss = self.loss_fn.forward(logits, yb)

                grad = self.loss_fn.backward(logits, yb)
                self.net.backward(grad)

                self.opt.step(self.net.params(), self.net.grads())

                train_losses.append(loss)
                y_true = yb if yb.ndim == 1 else np.argmax(yb, axis=1)
                train_accs.append(utils.accuracy(logits, y_true))

            history["train_loss"].append(float(np.mean(train_losses)))
            history["train_acc"].append(float(np.mean(train_accs)))

            if X_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if self.verbose:
                    print(f"Epoch {epoch:03d} | train_loss={history['train_loss'][-1]:.4f} "
                          f"val_loss={val_loss:.4f} train_acc={history['train_acc'][-1]:.3f} val_acc={val_acc:.3f}"
                    )

                if self.early_stopping:
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        patience_left = self.patience
                        best_params = {k: v.copy() for k, v in self.net.params().items()}
                    else:
                        patience_left -= 1
                        if patience_left == 0:
                            if self.verbose:
                                print("Early stopping activado. Restaurando mejores parámetros.")
                            for k, v in self.net.params().items():
                                v[:] = best_params[k]
                            break
            else:
                if self.verbose:
                    print(
                        f"Epoch {epoch:03d} | train_loss={history['train_loss'][-1]:.4f} "
                        f"train_acc={history['train_acc'][-1]:.3f}"
                    )

        return history
