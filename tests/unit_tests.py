import os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.layers import Dense, Sigmoid, ReLU
from src.losses import MSELoss, CrossEntropyLoss
from src.network import NeuralNetwork, Trainer
from src.optimizers import Adam
from src import utils


def rel_error(a, b, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + eps
    return float(num / den)


def scale_alpha(ana: np.ndarray, num: np.ndarray) -> float:
    aa = float(np.sum(ana * ana)) + 1e-12
    an = float(np.sum(ana * num))
    return an / aa


def numerical_gradient_check(seed=0, eps=1e-3, rtol=5e-2, alpha_tol=0.25):
    np.random.seed(seed)
    N, D, H, C = 5, 4, 6, 3
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.rand(N, C).astype(np.float32)

    model = NeuralNetwork([
        Dense(D, H, init="xavier"), Sigmoid(),
        Dense(H, C, init="xavier"), Sigmoid()
    ])
    loss = MSELoss()
    logits = model.forward(X, training=False)
    base_L = loss.forward(logits, y)
    dL = loss.backward(logits, y)
    model.backward(dL)
    analytic = model.grads()
    params = model.params()

    print(f">> Iniciando grad-check (MSE+Sigmoid). eps={eps:.1e}, rtol={rtol:.2f}")
    for k, W in params.items():
        g_num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = W[idx]

            W[idx] = old + eps
            L1 = loss.forward(model.forward(X, training=False), y)

            W[idx] = old - eps
            L2 = loss.forward(model.forward(X, training=False), y)

            W[idx] = old
            g_num[idx] = (L1 - L2) / (2.0 * eps)
            it.iternext()

        g_ana = analytic[k]
        rel = rel_error(g_num, g_ana)
        alpha = scale_alpha(g_ana, g_num)
        print(f"Grad-check {k:>6}: rel={rel:.3e} | alpha={alpha:.3f}")
        assert rel < rtol, f"Grad-check falló para {k}: rel={rel:.3e} >= {rtol:.2f}"
        assert abs(alpha - 1.0) < alpha_tol, (
            f"Posible desajuste de escala en {k}: alpha={alpha:.3f} (tolerancia ±{alpha_tol})"
        )

    print("Grad-check completado sin errores.\n")


def tiny_training_check():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])

    utils.set_seed(7)
    net = NeuralNetwork([
        Dense(2, 8, init="he"), ReLU(),
        Dense(8, 2, init="xavier")
    ])
    opt = Adam(lr=0.05)
    loss = CrossEntropyLoss()
    tr = Trainer(net, opt, loss, early_stopping=False, verbose=False)

    hist = tr.train(X, y, epochs=300, batch_size=4)
    assert hist["train_loss"][-1] < hist["train_loss"][0], "La pérdida no disminuyó"
    print("tiny_training_check → la pérdida disminuye.\n")


if __name__ == "__main__":
    print("UNIT TESTS: NEURAL NETWORK ENGINE\n")
    numerical_gradient_check()
    tiny_training_check()
    print("Todos los tests pasaron correctamente.")
