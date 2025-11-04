import numpy as np

from typing import Dict

class Optimizer:
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        raise NotImplementedError
    def zero_state(self):
        pass

class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m, self.v = {}, {}
        self.wd = weight_decay

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in grads: 
                continue
            g = grads[k].copy()
            if self.wd > 0.0 and k.startswith("W"):
                g += self.wd * params[k]
            m = self.m.setdefault(k, np.zeros_like(g))
            v = self.v.setdefault(k, np.zeros_like(g))
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * (g * g)
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            params[k][:] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.v = {}
    def step(self, params, grads):
        for k in params:
            if k not in grads: 
                continue
            g = grads[k].copy()
            if self.wd > 0.0 and k.startswith("W"):
                g += self.wd * params[k]
            v = self.v.setdefault(k, np.zeros_like(g))
            v[:] = self.momentum * v + g
            params[k][:] -= self.lr * v

class RMSProp(Optimizer):
    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr, self.beta, self.eps, self.wd = lr, beta, eps, weight_decay
        self.sq = {}
    def step(self, params, grads):
        for k in params:
            if k not in grads:
                continue
            g = grads[k].copy()
            if self.wd > 0.0 and k.startswith("W"):
                g += self.wd * params[k]
            s = self.sq.setdefault(k, np.zeros_like(g))
            s[:] = self.beta * s + (1 - self.beta) * (g * g)
            params[k][:] -= self.lr * g / (np.sqrt(s) + self.eps)
