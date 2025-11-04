import numpy as np

class Loss:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return float(0.5 * np.mean(diff * diff))
    def backward(self, y_pred, y_true):
        N = y_pred.shape[0]
        C = y_pred.shape[1] if y_pred.ndim == 2 else 1
        return (y_pred - y_true) / (N * C)

class CrossEntropyLoss(Loss):
    def forward(self, logits, y_true):
        if y_true.ndim == 1:
            y_true_oh = np.zeros_like(logits, dtype=logits.dtype)
            y_true_oh[np.arange(logits.shape[0]), y_true] = 1.0
        else:
            y_true_oh = y_true
        z = logits - logits.max(axis=1, keepdims=True)   
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_probs = z - logsumexp
        loss = - (y_true_oh * log_probs).sum(axis=1).mean() 
        self._y_true_oh = y_true_oh
        self._softmax = np.exp(log_probs)
        return float(loss)

    def backward(self, logits, y_true):
        if y_true.ndim == 1:
            y_true_oh = np.zeros_like(logits, dtype=logits.dtype)
            y_true_oh[np.arange(logits.shape[0]), y_true] = 1.0
        else:
            y_true_oh = y_true
        z = logits - logits.max(axis=1, keepdims=True)
        softmax = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return (softmax - y_true_oh) / logits.shape[0]
