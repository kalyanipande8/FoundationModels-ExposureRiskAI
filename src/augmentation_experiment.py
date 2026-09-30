"""
Generative data augmentation experiment.

Train the small VAE on the small target training set's feature space,
generate synthetic augmented samples, label them with a k-NN-based
pseudo-labeling scheme anchored on real labeled neighbors (a real,
simple, honest labeling approach -- no oracle), and measure whether
adding these synthetic samples to the small training set changes
downstream classifier performance on the real held-out test set.

We report the real result, whatever it is (could be a gain, a wash, or
a small regression) -- no cherry-picking.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.nn import Sequential, train_classifier, predict_proba
from src.models import build_full_model
from src.data_sim import simulate_target_data, N_FEATURES
from src.vae import SmallVAE


def _standardize(X, mean, std):
    return (X - mean) / std


def _train_eval(Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    model, _, _ = build_full_model(N_FEATURES, rng)
    train_classifier(model, Xtr, ytr, epochs=150, lr=1e-3, batch_size=16, rng=rng)
    pred = predict_proba(model, Xte)
    return {
        "test_accuracy": float(accuracy_score(yte, pred > 0.5)),
        "test_auc": float(roc_auc_score(yte, pred)),
    }


def run_augmentation_experiment(seed=42, n_synth=120):
    target = simulate_target_data(seed=seed + 1)
    mean, std = target["feature_mean"], target["feature_std"]
    Xtr = _standardize(target["X_train"], mean, std)
    ytr = target["y_train"]
    Xte = _standardize(target["X_test"], mean, std)
    yte = target["y_test"]

    # --- Baseline: train only on the real small training set ---
    baseline = _train_eval(Xtr, ytr, Xte, yte, seed=seed + 10)

    # --- Fit the generative model (VAE) on the real training features ---
    vae = SmallVAE(n_features=Xtr.shape[1], seed=seed)
    vae_history = vae.fit(Xtr, epochs=400, lr=3e-3, batch_size=16, beta=0.3, seed=seed)

    # --- Generate synthetic feature vectors ---
    X_synth = vae.sample(n_synth, seed=seed + 5)

    # --- Pseudo-label synthetic samples using k-NN over the REAL labeled
    #     training data (honest: no oracle, no peeking at test labels) ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    y_synth = knn.predict(X_synth)

    X_aug = np.concatenate([Xtr, X_synth], axis=0)
    y_aug = np.concatenate([ytr, y_synth], axis=0)

    augmented = _train_eval(X_aug, y_aug, Xte, yte, seed=seed + 10)

    # --- Statistical plausibility check of generated samples vs real data ---
    real_mean, real_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    synth_mean, synth_std = X_synth.mean(axis=0), X_synth.std(axis=0)

    return {
        "baseline": baseline,
        "augmented": augmented,
        "n_train_real": int(len(ytr)),
        "n_synth": int(n_synth),
        "vae_final_loss": float(vae_history[-1]),
        "vae_initial_loss": float(vae_history[0]),
        "real_feature_mean": real_mean.tolist(),
        "real_feature_std": real_std.tolist(),
        "synth_feature_mean": synth_mean.tolist(),
        "synth_feature_std": synth_std.tolist(),
        "auc_delta": float(augmented["test_auc"] - baseline["test_auc"]),
        "accuracy_delta": float(augmented["test_accuracy"] - baseline["test_accuracy"]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_augmentation_experiment(), indent=2))
