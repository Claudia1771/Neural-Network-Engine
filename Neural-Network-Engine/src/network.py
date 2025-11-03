import numpy as np
from src.layers import Dense
from src.losses import (
    cross_entropy, cross_entropy_grad,
    mse, mse_grad
)

class NeuralNetwork:
    def __init__(self, input_dim, layers_config, loss="cross_entropy"):
        self.loss_name = loss
        self.layers = []
        in_dim = input_dim
        for cfg in layers_config:
            layer = Dense(
                in_dim=in_dim,
                out_dim=cfg["units"],
                activation=cfg.get("activation", "sigmoid"),
                weight_init=cfg.get("weight_init", "xavier")
            )
            self.layers.append(layer)
            in_dim = cfg["units"]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss_and_gradients(self, X, y_true):
        y_pred = self.forward(X)
        if self.loss_name == "cross_entropy":
            loss = cross_entropy(y_true, y_pred)
            dA = cross_entropy_grad(y_true, y_pred)
        elif self.loss_name == "mse":
            loss = mse(y_true, y_pred)
            dA = mse_grad(y_true, y_pred)
        else:
            raise NotImplementedError
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return loss

    def get_params_and_grads(self):
        params_grads = []
        for layer in self.layers:
            params_grads.append((layer.W, layer.dW))
            params_grads.append((layer.b, layer.db))
        return params_grads

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
