"""Small runtime guards for split-aware experiments."""
from __future__ import annotations

import numpy as np


def split_reference(reference_idx, *, split_seed=42, fold=0, validation_fraction=0.125):
    """Training randomness must never change fitting/validation membership."""
    ref = np.asarray(reference_idx)
    if ref.ndim != 1 or ref.dtype.kind not in "iu" or len(np.unique(ref)) != len(ref):
        raise ValueError("Reference indices must be unique integers")
    if len(ref) < 3 or not 0 < validation_fraction < 1:
        raise ValueError("At least three reference observations and a valid fraction required")
    perm = np.random.default_rng(split_seed + fold).permutation(len(ref))
    n_val = max(1, int(len(ref) * validation_fraction))
    if len(ref) - n_val < 2:
        raise ValueError("At least two fitting observations required")
    return ref[perm[n_val:]], ref[perm[:n_val]]


def validate_three_way_split(train_idx, val_idx, test_idx, n_obs: int | None = None) -> None:
    """Require train/validation/test indices to be unique and pairwise disjoint."""
    tr = np.asarray(train_idx)
    va = np.asarray(val_idx)
    te = np.asarray(test_idx)
    for name, x in (("train", tr), ("validation", va), ("test", te)):
        if x.dtype.kind not in "iu":
            raise ValueError(f"{name} indices must have an integer dtype")
        if x.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional")
        if len(np.unique(x)) != len(x):
            raise ValueError(f"{name} indices contain duplicates")
        if n_obs is not None and (np.any(x < 0) or np.any(x >= n_obs)):
            raise ValueError(f"{name} indices are outside [0, {n_obs})")
    if np.intersect1d(tr, va).size:
        raise ValueError("train and validation splits overlap")
    if np.intersect1d(tr, te).size:
        raise ValueError("train and test splits overlap")
    if np.intersect1d(va, te).size:
        raise ValueError("validation and test splits overlap")
