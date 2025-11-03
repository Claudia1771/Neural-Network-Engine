import numpy as np

class Optimizer:
    def step(self, params_and_grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params_and_grads):
        for p, g in params_and_grads:
            p -= self.lr * g

class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params_and_grads):
        self.t += 1
        for idx, (p, g) in enumerate(params_and_grads):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(g)
                self.v[idx] = np.zeros_like(g)
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
