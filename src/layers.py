import numpy as np
from typing import Dict

class Layer:
    """Clase base para todas las capas de la red neuronal."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagación hacia adelante."""
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás."""
        raise NotImplementedError
    
    def params(self) -> Dict[str, np.ndarray]:
        """Devuelve los parámetros entrenables de la capa."""
        return {}
    
    def grads(self) -> Dict[str, np.ndarray]:
        """Devuelve los gradientes de los parámetros."""
        return {}

# ---- Inicializaciones ----

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Inicialización Xavier (Glorot) para pesos."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Inicialización He para capas ReLU."""
    std = np.sqrt(2.0 / fan_in)
    return (np.random.randn(fan_in, fan_out) * std).astype(np.float32)

# ---- Capas ----

class Dense(Layer):
    """Capa totalmente conectada (Fully Connected)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.W = he_init(in_features, out_features) if init == "he" else xavier_init(in_features, out_features)
        self.b = np.zeros((1, out_features), dtype=np.float32) if bias else None
        self._cache_x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagación hacia adelante."""
        self._cache_x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás."""
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

# ---- Activaciones ----

class Sigmoid(Layer):
    """Función de activación Sigmoid."""

    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)

class Tanh(Layer):
    """Función de activación Tanh."""

    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)

class ReLU(Layer):
    """Función de activación ReLU."""

    def forward(self, x, training=True):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask

class Softmax(Layer):
    """Función de activación Softmax."""

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
    
# ---- Regularización ----

class Dropout(Layer):
    """Capa de Dropout."""
    def __init__(self, p: float = 0.5):
        assert 0 <= p < 1, "p debe estar en [0,1)"
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
