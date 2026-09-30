"""
Real transfer-learning experiment.

1. Pretrain an encoder+head on the large, diverse synthetic pretraining
   set (`simulate_pretrain_data`).
2. Take ONLY the encoder weights, attach a fresh head, and fine-tune on
   the small target task (`simulate_target_data`) -- this is "transfer".
3. Train an architecturally-identical model from scratch (randomly
   initialized) on the SAME small target training set only.
4. Compare test-set accuracy/AUC honestly. No numbers are hand-picked;
   whatever the two numbers are is what gets reported.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.nn import Sequential, train_classifier, predict_proba
from src.models import build_full_model, build_head, clone_layers
from src.data_sim import simulate_pretrain_data, simulate_target_data, N_FEATURES


def _standardize(X, mean, std):
    return (X - mean) / std


def pretrain_encoder(seed=0, epochs=60):
    rng = np.random.default_rng(seed)
    X, y = simulate_pretrain_data(seed=seed)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xz = _standardize(X, mean, std)
    model, encoder, head = build_full_model(N_FEATURES, rng)
    history = train_classifier(model, Xz, y, epochs=epochs, lr=2e-3, batch_size=64, rng=rng)
    pretrain_pred = predict_proba(model, Xz)
    pretrain_acc = accuracy_score(y, pretrain_pred > 0.5)
    return {
        "encoder": encoder,
        "pretrain_loss_curve": history,
        "pretrain_train_accuracy": float(pretrain_acc),
        "feature_mean": mean,
        "feature_std": std,
    }


def run_transfer_vs_scratch(seed=42, fine_tune_encoder=False, epochs=150):
    pre = pretrain_encoder(seed=seed)
    target = simulate_target_data(seed=seed + 1)
    mean, std = target["feature_mean"], target["feature_std"]
    Xtr = _standardize(target["X_train"], mean, std)
    ytr = target["y_train"]
    Xte = _standardize(target["X_test"], mean, std)
    yte = target["y_test"]

    rng = np.random.default_rng(seed + 2)

    # --- Transfer model: pretrained encoder + fresh head ---
    transfer_encoder = clone_layers(pre["encoder"])
    transfer_head = build_head(rng)
    transfer_model = Sequential(transfer_encoder + transfer_head)
    if not fine_tune_encoder:
        transfer_model.set_trainable(False, layers=transfer_encoder)
    transfer_history = train_classifier(
        transfer_model, Xtr, ytr, epochs=epochs, lr=1e-3, batch_size=16, rng=rng
    )
    transfer_test_pred = predict_proba(transfer_model, Xte)
    transfer_acc = accuracy_score(yte, transfer_test_pred > 0.5)
    transfer_auc = roc_auc_score(yte, transfer_test_pred)

    # --- From-scratch model: identical architecture, random init, trained
    #     only on the small target set ---
    rng2 = np.random.default_rng(seed + 3)
    scratch_model, _, _ = build_full_model(N_FEATURES, rng2)
    scratch_history = train_classifier(
        scratch_model, Xtr, ytr, epochs=epochs, lr=1e-3, batch_size=16, rng=rng2
    )
    scratch_test_pred = predict_proba(scratch_model, Xte)
    scratch_acc = accuracy_score(yte, scratch_test_pred > 0.5)
    scratch_auc = roc_auc_score(yte, scratch_test_pred)

    return {
        "pretrain_train_accuracy": pre["pretrain_train_accuracy"],
        "transfer": {"test_accuracy": float(transfer_acc), "test_auc": float(transfer_auc)},
        "scratch": {"test_accuracy": float(scratch_acc), "test_auc": float(scratch_auc)},
        "n_train_target": int(len(ytr)),
        "n_test_target": int(len(yte)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_transfer_vs_scratch(), indent=2))
