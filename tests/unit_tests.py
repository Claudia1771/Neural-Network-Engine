import numpy as np
from src.network import NeuralNetwork

def test_simple_forward():
    np.random.seed(0)
    net = NeuralNetwork(
        input_dim=4,
        layers_config=[
            {"units": 5, "activation": "sigmoid"},
            {"units": 3, "activation": "softmax"},
        ],
        loss="cross_entropy"
    )
    X = np.random.randn(2, 4).astype(np.float32)
    y = np.array([0, 2])
    y_oh = np.zeros((2, 3))
    y_oh[np.arange(2), y] = 1
    loss = net.compute_loss_and_gradients(X, y_oh)
    assert loss > 0
    print("test_simple_forward OK")

if __name__ == "__main__":
    test_simple_forward()
