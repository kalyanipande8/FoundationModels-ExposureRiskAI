"""
Synthetic air-quality / exposure-risk data simulation.

Two datasets are generated:

1. `simulate_pretrain_data` — a LARGE, DIVERSE synthetic dataset spanning
   many different "regions" (each with its own randomly perturbed
   dose-response coefficients between pollutants/weather and exposure
   risk). This stands in for a broad pretraining corpus: the encoder that
   is pretrained on this sees a wide variety of pollutant/weather ->
   risk relationships, similar in spirit to how a foundation model sees
   broad data before being adapted to a narrow downstream task.

2. `simulate_target_data` — a SMALL dataset from one specific held-out
   "region" with its own coefficient vector (drawn from the same family
   but never seen during pretraining) and added feature noise. This is
   the realistic small target task: few labeled samples, the situation
   where transfer learning is expected to actually help.

All relationships are real (deterministic function of features + noise),
computed with numpy — nothing here is a live model call.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "pm25", "pm10", "no2", "o3", "co", "temperature", "humidity", "wind_speed",
]
N_FEATURES = len(FEATURE_NAMES)


def _raw_features(n, rng):
    pm25 = rng.gamma(shape=2.0, scale=15.0, size=n)          # ug/m3
    pm10 = pm25 * rng.uniform(1.3, 1.8, size=n) + rng.normal(0, 5, n)
    no2 = rng.gamma(shape=2.5, scale=12.0, size=n)           # ppb
    o3 = rng.gamma(shape=3.0, scale=15.0, size=n)            # ppb
    co = rng.gamma(shape=2.0, scale=0.4, size=n)             # ppm
    temperature = rng.normal(22, 8, size=n)                  # C
    humidity = np.clip(rng.normal(55, 18, size=n), 5, 100)   # %
    wind_speed = np.clip(rng.gamma(2.0, 2.0, size=n), 0.1, None)  # m/s
    X = np.stack([pm25, pm10, no2, o3, co, temperature, humidity, wind_speed], axis=1)
    return X


def _standardize_columns(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def _risk_score(Xz, coefs, bias, noise_scale, rng):
    """Nonlinear dose-response function -> continuous latent risk score."""
    linear = Xz @ coefs + bias
    interaction = 0.5 * Xz[:, 0] * Xz[:, 5] - 0.3 * Xz[:, 7] * Xz[:, 2]  # pm25*temp, wind*no2
    nonlinear = 0.4 * np.tanh(Xz[:, 0]) + 0.3 * np.square(np.clip(Xz[:, 2], -3, 3)) * 0.3
    score = linear + interaction + nonlinear + rng.normal(0, noise_scale, size=Xz.shape[0])
    return score


def _base_coefs():
    # Base "dose-response" direction: PM2.5, PM10, NO2, O3, CO raise risk;
    # wind speed lowers it (dispersion); temperature/humidity mild effects.
    return np.array([0.9, 0.4, 0.7, 0.5, 0.6, 0.15, 0.1, -0.6])


def simulate_pretrain_data(n_regions=40, n_per_region=150, seed=0):
    """Broad, diverse pretraining set across many synthetic regions."""
    rng = np.random.default_rng(seed)
    base = _base_coefs()
    all_X, all_y = [], []
    for r in range(n_regions):
        n = n_per_region
        region_rng = np.random.default_rng(seed * 1000 + r)
        X = _raw_features(n, region_rng)
        Xz, _, _ = _standardize_columns(X)
        # Each region perturbs the dose-response coefficients -- this is
        # what makes pretraining "broad": many related but distinct tasks.
        coefs = base * region_rng.uniform(0.6, 1.4, size=N_FEATURES)
        bias = region_rng.normal(0, 0.3)
        score = _risk_score(Xz, coefs, bias, noise_scale=0.6, rng=region_rng)
        y = (score > np.median(score)).astype(float)
        all_X.append(X)
        all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def simulate_target_data(n_train=60, n_test=150, seed=777):
    """
    One small, specific target "city" task, held out from pretraining
    coefficient draws (different seed stream and a distinct systematic
    shift), representing the real downstream exposure-risk task we
    actually care about.
    """
    rng = np.random.default_rng(seed)
    base = _base_coefs()
    # A specific target-domain coefficient vector: related to the pretraining
    # family but with its own systematic shift (domain shift), e.g. a city
    # with unusually strong PM2.5/NO2 sensitivity and weaker wind mitigation.
    coefs = base * np.array([1.3, 0.9, 1.2, 0.8, 0.7, 0.5, 0.6, 0.4])
    bias = 0.15

    n_total = n_train + n_test
    X = _raw_features(n_total, rng)
    Xz, mean, std = _standardize_columns(X)
    score = _risk_score(Xz, coefs, bias, noise_scale=0.75, rng=rng)
    y = (score > np.median(score)).astype(float)

    idx = rng.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "feature_mean": mean, "feature_std": std,
    }
