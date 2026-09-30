"""
Minimal, dependency-free feedforward neural network implementation
(NumPy only) with manual forward/backward passes and an Adam optimizer.

Why not PyTorch / a real transformer library?
-----------------------------------------------
This machine does not have PyTorch installed and has very little free
disk space (~1.5GB), so installing a multi-hundred-MB deep learning
framework was judged unsafe. Instead, this module implements a real,
from-scratch, gradient-descent-trained multilayer perceptron (MLP) that
supports the one feature that off-the-shelf sklearn MLPs do not expose
easily: freezing an arbitrary subset of layers so a "pretrained encoder"
can be genuinely reused (frozen or fine-tuned) for transfer learning.

Everything below is real: real matrix multiplies, real backprop,
real Adam updates. No stub / mocked training loop.
"""
from __future__ import annotations
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


class Dense:
    """A single fully connected layer with its own Adam moment buffers."""

    def __init__(self, n_in, n_out, activation="relu", rng=None, trainable=True):
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation
        self.trainable = trainable
        # Adam moment buffers
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0
        # caches
        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        z = x @ self.W + self.b
        self._z = z
        if self.activation == "relu":
            return relu(z)
        elif self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "linear":
            return z
        else:
            raise ValueError(self.activation)

    def backward(self, dout):
        if self.activation == "relu":
            dz = dout * relu_grad(self._z)
        elif self.activation == "linear":
            dz = dout
        elif self.activation == "sigmoid":
            s = sigmoid(self._z)
            dz = dout * s * (1 - s)
        else:
            raise ValueError(self.activation)
        dW = self._x.T @ dz / self._x.shape[0]
        db = dz.mean(axis=0)
        dx = dz @ self.W.T
        self._dW, self._db = dW, db
        return dx

    def adam_step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        if not self.trainable:
            return
        self.t += 1
        for p, dp, m_name, v_name in (
            (self.W, self._dW, "mW", "vW"),
            (self.b, self._db, "mb", "vb"),
        ):
            m = getattr(self, m_name)
            v = getattr(self, v_name)
            m[:] = beta1 * m + (1 - beta1) * dp
            v[:] = beta2 * v + (1 - beta2) * (dp ** 2)
            mhat = m / (1 - beta1 ** self.t)
            vhat = v / (1 - beta2 ** self.t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, lr=1e-3):
        for layer in self.layers:
            layer.adam_step(lr=lr)

    def set_trainable(self, trainable: bool, layers=None):
        target = self.layers if layers is None else layers
        for layer in target:
            layer.trainable = trainable


def bce_loss_and_grad(y_pred, y_true):
    """Binary cross-entropy loss (y_pred already sigmoid-activated)."""
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_true = y_true.reshape(-1, 1)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])
    return loss, grad


def train_classifier(model: Sequential, X, y, epochs=200, lr=1e-3, batch_size=32, rng=None, verbose=False):
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        Xs, ys = X[idx], y[idx]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            if len(xb) == 0:
                continue
            pred = model.forward(xb)
            loss, grad = bce_loss_and_grad(pred, yb)
            model.backward(grad)
            model.step(lr=lr)
            epoch_loss += loss
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch}: loss={history[-1]:.4f}")
    return history


def predict_proba(model: Sequential, X):
    return model.forward(X).ravel()
