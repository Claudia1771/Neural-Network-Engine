import numpy as np

from typing import Dict

class Layer:
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def params(self) -> Dict[str, np.ndarray]:
        return {}
    def grads(self) -> Dict[str, np.ndarray]:
        return {}

# Inicializaciones
def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return (np.random.randn(fan_in, fan_out) * std).astype(np.float32)

# Capa totalmente conectada
class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.W = he_init(in_features, out_features) if init == "he" else xavier_init(in_features, out_features)
        self.b = np.zeros((1, out_features), dtype=np.float32) if bias else None
        self._cache_x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache_x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self._cache_x  
        self.dW = x.T @ grad_output
        if self.b is not None:
            self.db = grad_output.sum(axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

    def params(self):
        p = {"W": self.W}
        if self.b is not None:
            p["b"] = self.b
        return p

    def grads(self):
        g = {"W": self.dW}
        if self.b is not None:
            g["b"] = self.db
        return g

# Activaciones
class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)

class Tanh(Layer):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)

class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    def backward(self, grad_output):
        return grad_output * self.mask

class Softmax(Layer):
    def forward(self, x, training=True):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out
    def backward(self, grad_output):
        N, C = grad_output.shape
        grad_input = np.zeros_like(grad_output)
        for i in range(N):
            s = self.out[i].reshape(-1, 1)
            J = np.diagflat(s) - s @ s.T
            grad_input[i] = J @ grad_output[i]
        return grad_input

# Dropout
class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        assert 0 <= p < 1, "p en [0,1)"
        self.p = p
        self.mask = None
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32)
            return x * self.mask / (1.0 - self.p)
        else:
            return x
    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.p)
