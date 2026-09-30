"""Model factory helpers: builds the shared encoder + classification head
architecture used for both pretraining and the downstream target task."""
from __future__ import annotations
import numpy as np
import copy
from src.nn import Dense, Sequential

ENCODER_HIDDEN = (32, 16)
HEAD_HIDDEN = 8


def build_encoder(n_features, rng):
    return [
        Dense(n_features, ENCODER_HIDDEN[0], activation="relu", rng=rng),
        Dense(ENCODER_HIDDEN[0], ENCODER_HIDDEN[1], activation="relu", rng=rng),
    ]


def build_head(rng, in_dim=ENCODER_HIDDEN[1]):
    return [
        Dense(in_dim, HEAD_HIDDEN, activation="relu", rng=rng),
        Dense(HEAD_HIDDEN, 1, activation="sigmoid", rng=rng),
    ]


def build_full_model(n_features, rng):
    encoder = build_encoder(n_features, rng)
    head = build_head(rng)
    return Sequential(encoder + head), encoder, head


def clone_layers(layers):
    return copy.deepcopy(layers)
