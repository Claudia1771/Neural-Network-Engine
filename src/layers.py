import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(a):
    return a * (1.0 - a)

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim))

class Dense:
    def __init__(self, in_dim, out_dim, activation="sigmoid", weight_init="xavier"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = activation

        if weight_init == "xavier":
            self.W = xavier_init(in_dim, out_dim)
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))

        self.Z = None
        self.A_in = None
        self.A_out = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, A_in):
        self.A_in = A_in
        self.Z = A_in @ self.W + self.b
        if self.activation_name == "sigmoid":
            self.A_out = sigmoid(self.Z)
        elif self.activation_name == "softmax":
            self.A_out = softmax(self.Z)
        else:
            raise NotImplementedError("Solo sigmoid y softmax")
        return self.A_out

    def backward(self, dA_out):
        if self.activation_name == "sigmoid":
            dZ = dA_out * sigmoid_grad(self.A_out)
        elif self.activation_name == "softmax":
            dZ = dA_out
        else:
            raise NotImplementedError
        self.dW = self.A_in.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dA_in = dZ @ self.W.T
        return dA_in
