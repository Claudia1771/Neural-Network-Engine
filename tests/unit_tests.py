# Este fichero ejecuta pruebas unitarias para verificar el funcionamiento de los métodos principales del motor:
# capas, activaciones, pérdidas, optimizadores, regularización (dropout), schedulers, red secuencial, entrenamiento
# (incluyendo early stopping) y utilidades (one-hot, batches y particionado).


import unittest
import numpy as np
import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.network import NeuralNetwork, step_decay, cosine_annealing, Trainer
from src.layers import Dense, Dropout, Sigmoid, Tanh, ReLU, Softmax
from src.losses import MSELoss, CrossEntropyLoss
from src.optimizers import SGD, RMSProp, Adam
from src import utils

class TestNeuralNetworkEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        self.nn = NeuralNetwork([
            Dense(4, 5, init="he"),
            ReLU(),
            Dense(5, 3),
            Softmax()
        ])

        self.X = np.random.randn(10, 4).astype(np.float32)
        self.y = np.zeros((10, 3), dtype=np.float32)
        self.y[np.arange(10), np.random.randint(0, 3, 10)] = 1.0

    # Dense.forward(), Dense.backward(), Dense.params(), Dense.grads()
    def test_dense_forward_backward_shapes(self):
        dense = Dense(4, 5, init="he")
        x = np.random.randn(7, 4).astype(np.float32)
        out = dense.forward(x, training=True)
        self.assertEqual(out.shape, (7, 5))

        grad_out = np.random.randn(7, 5).astype(np.float32)
        grad_in = dense.backward(grad_out)
        self.assertEqual(grad_in.shape, (7, 4))
        self.assertEqual(dense.grads()["W"].shape, dense.params()["W"].shape)
        if "b" in dense.params():
            self.assertEqual(dense.grads()["b"].shape, dense.params()["b"].shape)

    # Sigmoid.forward(), Sigmoid.backward()
    def test_sigmoid(self):
        x = np.array([-1, 0, 1], dtype=np.float32)
        sig = Sigmoid()
        s = sig.forward(x)
        self.assertTrue(np.allclose(s, [0.2689, 0.5, 0.7310], atol=1e-3))

        grad_out = np.ones_like(s, dtype=np.float32)
        grad_in = sig.backward(grad_out)
        self.assertEqual(grad_in.shape, x.shape)
        self.assertTrue(np.isfinite(grad_in).all())

    # Tanh.forward(), Tanh.backward()
    def test_tanh(self):
        x = np.array([-1, 0, 1], dtype=np.float32)
        t_layer = Tanh()
        t = t_layer.forward(x)
        self.assertTrue(np.allclose(t, [-0.7616, 0, 0.7616], atol=1e-3))

        grad_out = np.ones_like(t, dtype=np.float32)
        grad_in = t_layer.backward(grad_out)
        self.assertEqual(grad_in.shape, x.shape)

    # ReLU.forward(), ReLU.backward()
    def test_relu(self):
        x = np.array([-2, 0, 3], dtype=np.float32)
        relu = ReLU()
        r = relu.forward(x)
        self.assertTrue(np.all(r >= 0))

        grad_out = np.ones_like(r, dtype=np.float32)
        grad_in = relu.backward(grad_out)
        self.assertEqual(grad_in.shape, x.shape)
        self.assertEqual(float(grad_in[0]), 0.0)

    # Softmax.forward(), Softmax.backward()
    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        sm = Softmax()
        s = sm.forward(x)
        self.assertTrue(np.isclose(np.sum(s), 1.0))

        grad_out = np.random.randn(*s.shape).astype(np.float32)
        grad_in = sm.backward(grad_out)
        self.assertEqual(grad_in.shape, x.shape)

    # MSELoss.forward(), MSELoss.backward()
    def test_mse_loss(self):
        mse = MSELoss()
        y_true = np.array([[1, 0]], dtype=np.float32)
        y_pred = np.array([[0.8, 0.2]], dtype=np.float32)

        loss = mse.forward(y_pred, y_true)
        self.assertGreater(loss, 0.0)

        grad = mse.backward(y_pred, y_true)
        self.assertEqual(grad.shape, y_pred.shape)
        self.assertTrue(np.isfinite(grad).all())

    # CrossEntropyLoss.forward(), CrossEntropyLoss.backward()
    def test_cross_entropy_loss_onehot(self):
        ce = CrossEntropyLoss()
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        logits = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)

        loss = ce.forward(logits, y_true)
        self.assertGreater(loss, 0.0)

        grad = ce.backward(logits, y_true)
        self.assertEqual(grad.shape, logits.shape)
        self.assertTrue(np.isfinite(grad).all())

    # CrossEntropyLoss.forward(), CrossEntropyLoss.backward()
    def test_cross_entropy_loss_labels(self):
        ce = CrossEntropyLoss()
        y_true = np.array([1], dtype=np.int64)
        logits = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)

        loss = ce.forward(logits, y_true)
        self.assertGreater(loss, 0.0)

        grad = ce.backward(logits, y_true)
        self.assertEqual(grad.shape, logits.shape)
        self.assertTrue(np.isfinite(grad).all())

    # Dropout.forward(), Dropout.backward()
    def test_dropout_training_and_eval(self):
        d = Dropout(0.5)
        x = np.ones((10, 10), dtype=np.float32)

        out_train = d.forward(x, training=True)
        self.assertTrue(np.any(out_train == 0))
        self.assertTrue(np.all(out_train <= 2))

        out_eval = d.forward(x, training=False)
        self.assertTrue(np.allclose(out_eval, x))

        grad_out = np.ones_like(x, dtype=np.float32)
        grad_in = d.backward(grad_out)
        self.assertEqual(grad_in.shape, x.shape)

    # SGD.step()
    def test_sgd_update(self):
        sgd = SGD(lr=0.01, momentum=0.9)
        params = {"W_0": np.ones((2, 2), dtype=np.float32)}
        grads = {"W_0": np.ones((2, 2), dtype=np.float32)}
        sgd.step(params, grads)
        self.assertFalse(np.allclose(params["W_0"], np.ones((2, 2), dtype=np.float32)))

    # RMSProp.step()
    def test_rmsprop_update(self):
        rms = RMSProp(lr=0.01)
        params = {"W_0": np.ones((2, 2), dtype=np.float32)}
        grads = {"W_0": np.ones((2, 2), dtype=np.float32)}
        rms.step(params, grads)
        self.assertFalse(np.allclose(params["W_0"], np.ones((2, 2), dtype=np.float32)))

    # Adam.step()
    def test_adam_update(self):
        adam = Adam(lr=0.01)
        params = {"W_0": np.ones((2, 2), dtype=np.float32)}
        grads = {"W_0": np.ones((2, 2), dtype=np.float32)}
        adam.step(params, grads)
        self.assertFalse(np.allclose(params["W_0"], np.ones((2, 2), dtype=np.float32)))

    # Adam.step()
    def test_weight_decay_applies_only_to_weights(self):
        adam = Adam(lr=0.0, weight_decay=0.1)
        params = {
            "W_0": np.ones((2, 2), dtype=np.float32),
            "b_0": np.ones((1, 2), dtype=np.float32),
        }
        grads = {
            "W_0": np.zeros((2, 2), dtype=np.float32),
            "b_0": np.zeros((1, 2), dtype=np.float32),
        }
        adam.step(params, grads)
        self.assertTrue(np.allclose(params["b_0"], np.ones((1, 2), dtype=np.float32)))

    # step_decay()
    def test_step_decay(self):
        lr_values = [step_decay(epoch, base_lr=0.1, drop=0.5, every=2) for epoch in range(6)]
        self.assertTrue(all(lr_values[i] >= lr_values[i + 1] for i in range(5)))

    # cosine_annealing()
    def test_cosine_annealing(self):
        lr0 = cosine_annealing(0, base_lr=0.1, max_epochs=10)
        lr5 = cosine_annealing(5, base_lr=0.1, max_epochs=10)
        self.assertTrue(lr5 < lr0)

    # NeuralNetwork.forward(), NeuralNetwork.backward(), NeuralNetwork.params(), NeuralNetwork.grads()
    def test_network_forward_backward_and_grad_shapes(self):
        logits = self.nn.forward(self.X, training=True)
        self.assertEqual(logits.shape, (10, 3))
        self.assertTrue(np.isfinite(logits).all())

        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, self.y)
        self.assertTrue(np.isfinite(loss))

        grad = loss_fn.backward(logits, self.y)
        self.nn.backward(grad)

        params = self.nn.params()
        grads = self.nn.grads()
        self.assertTrue(len(params) > 0)
        self.assertTrue(len(grads) > 0)

        for k in grads:
            self.assertIn(k, params)
            self.assertEqual(grads[k].shape, params[k].shape)
            self.assertTrue(np.isfinite(grads[k]).all())

    # Trainer.train()
    def test_training_history_keys(self):
        optimizer = Adam(lr=0.01)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(self.nn, optimizer, loss_fn, early_stopping=False, verbose=False)
        history = trainer.train(self.X, self.y, epochs=3, batch_size=5)

        self.assertIn("train_loss", history)
        self.assertIn("train_acc", history)
        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["train_acc"]), 3)

    # Trainer.train()
    def test_early_stopping_triggers(self):
        X_val = self.X.copy()
        y_val = self.y.copy()

        optimizer = Adam(lr=0.0)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(
            self.nn, optimizer, loss_fn,
            early_stopping=True, patience=2,
            verbose=False
        )
        history = trainer.train(self.X, self.y, X_val=X_val, y_val=y_val, epochs=10, batch_size=5)
        self.assertLess(len(history["train_loss"]), 10)

    # utils.train_val_test_split()
    def test_data_split(self):
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = utils.train_val_test_split(
            self.X, self.y, train_size=0.7, val_size=0.15, test_size=0.15, seed=42, shuffle=True
        )
        total = len(X_tr) + len(X_val) + len(X_te)
        self.assertEqual(total, len(self.X))
        self.assertEqual(X_tr.shape[1], self.X.shape[1])

    # utils.batch_iterator()
    def test_batch_iterator_covers_all_samples(self):
        X = np.random.randn(23, 4).astype(np.float32)
        y = np.random.randn(23, 3).astype(np.float32)

        seen = 0
        for xb, yb in utils.batch_iterator(X, y, batch_size=7, shuffle=False):
            self.assertEqual(xb.shape[0], yb.shape[0])
            seen += xb.shape[0]
        self.assertEqual(seen, 23)

    # utils.one_hot()
    def test_one_hot(self):
        y = np.array([0, 2, 1, 2], dtype=np.int64)
        oh = utils.one_hot(y, num_classes=3)
        self.assertEqual(oh.shape, (4, 3))
        self.assertTrue(np.allclose(oh.sum(axis=1), 1.0))


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestNeuralNetworkEngine))

    if result.wasSuccessful():
        print("\nTodas las pruebas unitarias se han superado correctamente. "
              "Los métodos principales del motor han sido validados.\n")
    else:
        print("\nSe han detectado fallos en las pruebas unitarias. "
              "Revisa los errores anteriores para localizar el método afectado.\n")

