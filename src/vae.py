"""
A small Variational Autoencoder (VAE), implemented from scratch in NumPy
(forward pass, reparameterization trick, ELBO loss, manual backprop,
Adam updates). This is the "generative model" used for (a) data
augmentation of the small target task and (b) synthetic scenario
generation.

This is a real generative model: it learns a latent Gaussian
distribution over the standardized feature space and can sample new,
never-seen synthetic feature vectors from it.
"""
from __future__ import annotations
import numpy as np


def _init_layer(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    W = rng.uniform(-limit, limit, size=(n_in, n_out))
    b = np.zeros(n_out)
    return W, b


class AdamParam:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def update(self, p, grad, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
        mhat = self.m / (1 - beta1 ** self.t)
        vhat = self.v / (1 - beta2 ** self.t)
        return p - lr * mhat / (np.sqrt(vhat) + eps)


class SmallVAE:
    """Encoder: x -> h -> (mu, logvar). Decoder: z -> h -> x_hat."""

    def __init__(self, n_features, hidden=16, latent=4, seed=0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden = hidden
        self.latent = latent
        self.rng = rng

        self.W1, self.b1 = _init_layer(n_features, hidden, rng)
        self.Wmu, self.bmu = _init_layer(hidden, latent, rng)
        self.Wlv, self.blv = _init_layer(hidden, latent, rng)
        self.Wd1, self.bd1 = _init_layer(latent, hidden, rng)
        self.Wd2, self.bd2 = _init_layer(hidden, n_features, rng)

        params = ["W1", "b1", "Wmu", "bmu", "Wlv", "blv", "Wd1", "bd1", "Wd2", "bd2"]
        self._opt = {name: AdamParam(getattr(self, name).shape) for name in params}
        self._param_names = params

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def encode(self, x):
        h = self._relu(x @ self.W1 + self.b1)
        mu = h @ self.Wmu + self.bmu
        logvar = h @ self.Wlv + self.blv
        return h, mu, logvar

    def decode(self, z):
        hd = self._relu(z @ self.Wd1 + self.bd1)
        x_hat = hd @ self.Wd2 + self.bd2
        return hd, x_hat

    def forward(self, x, rng=None):
        rng = rng or self.rng
        h, mu, logvar = self.encode(x)
        std = np.exp(0.5 * logvar)
        eps = rng.normal(size=mu.shape)
        z = mu + eps * std
        hd, x_hat = self.decode(z)
        return dict(h=h, mu=mu, logvar=logvar, std=std, eps=eps, z=z, hd=hd, x_hat=x_hat)

    def loss(self, x, out, beta=0.5):
        recon = np.mean(np.sum((out["x_hat"] - x) ** 2, axis=1))
        kl = -0.5 * np.mean(np.sum(1 + out["logvar"] - out["mu"] ** 2 - np.exp(out["logvar"]), axis=1))
        return recon + beta * kl, recon, kl

    def train_step(self, x, lr=2e-3, beta=0.5):
        n = x.shape[0]
        out = self.forward(x)
        loss, recon, kl = self.loss(x, out, beta=beta)

        # --- gradients (manual backprop through the whole VAE) ---
        d_xhat = 2.0 * (out["x_hat"] - x) / n                       # dL_recon/dx_hat
        d_Wd2 = out["hd"].T @ d_xhat
        d_bd2 = d_xhat.sum(axis=0)
        d_hd = d_xhat @ self.Wd2.T
        d_hd_pre = d_hd * (out["hd"] > 0)
        d_Wd1 = out["z"].T @ d_hd_pre
        d_bd1 = d_hd_pre.sum(axis=0)
        d_z = d_hd_pre @ self.Wd1.T

        # reparameterization: z = mu + eps*std, std = exp(0.5*logvar)
        d_mu_recon = d_z.copy()
        d_std = d_z * out["eps"]
        d_logvar_recon = d_std * 0.5 * out["std"]

        # KL term gradients
        d_mu_kl = beta * out["mu"] / n
        d_logvar_kl = beta * 0.5 * (np.exp(out["logvar"]) - 1) / n

        d_mu = d_mu_recon + d_mu_kl
        d_logvar = d_logvar_recon + d_logvar_kl

        d_Wmu = out["h"].T @ d_mu
        d_bmu = d_mu.sum(axis=0)
        d_Wlv = out["h"].T @ d_logvar
        d_blv = d_logvar.sum(axis=0)
        d_h = d_mu @ self.Wmu.T + d_logvar @ self.Wlv.T
        d_h_pre = d_h * (out["h"] > 0)
        d_W1 = x.T @ d_h_pre
        d_b1 = d_h_pre.sum(axis=0)

        grads = dict(W1=d_W1, b1=d_b1, Wmu=d_Wmu, bmu=d_bmu, Wlv=d_Wlv, blv=d_blv,
                     Wd1=d_Wd1, bd1=d_bd1, Wd2=d_Wd2, bd2=d_bd2)
        for name in self._param_names:
            new_val = self._opt[name].update(getattr(self, name), grads[name], lr=lr)
            setattr(self, name, new_val)
        return loss, recon, kl

    def fit(self, X, epochs=300, lr=2e-3, batch_size=32, beta=0.5, seed=1, verbose=False):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = []
        for epoch in range(epochs):
            idx = rng.permutation(n)
            Xs = X[idx]
            tot = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                if len(xb) < 2:
                    continue
                loss, recon, kl = self.train_step(xb, lr=lr, beta=beta)
                tot += loss
                nb += 1
            history.append(tot / max(nb, 1))
            if verbose and epoch % 50 == 0:
                print(f"vae epoch {epoch} loss {history[-1]:.4f}")
        return history

    def sample(self, n, seed=123, z_mean_shift=None, z_std_scale=1.0):
        """Draw n synthetic feature vectors from the learned generative model.

        z_mean_shift / z_std_scale let us push the latent prior for
        scenario modeling (e.g. simulate a shifted future distribution).
        """
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(n, self.latent)) * z_std_scale
        if z_mean_shift is not None:
            z = z + z_mean_shift
        _, x_gen = self.decode(z)
        return x_gen
