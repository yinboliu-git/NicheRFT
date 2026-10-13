"""Retrieval utilities for retrieval-initialized flow matching.

Unlike the legacy MemoryBank helper, this module never mutates caller-owned
feature arrays and returns both the protein prior and the matched RNA latent
reference needed to form a transcriptomic displacement condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


@dataclass
class FlowRetrievalResult:
    indices: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    protein_prior: np.ndarray
    reference_feature: np.ndarray
    mean_similarity: np.ndarray


class FlowMemoryBank:
    """Exact cosine-similarity bank with protein and RNA-reference outputs."""

    def __init__(self, use_faiss: bool = True, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.use_faiss = bool(use_faiss and faiss is not None)
        self.temperature = float(temperature)
        self.features: Optional[np.ndarray] = None
        self.proteins: Optional[np.ndarray] = None
        self.index = None
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, features: np.ndarray, proteins: np.ndarray) -> None:
        feats = l2_normalize_np(features)
        proteins = np.asarray(proteins, dtype=np.float32).copy()
        if len(feats) != len(proteins):
            raise ValueError("features and proteins must have the same number of rows")
        self.features = feats
        self.proteins = proteins
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(feats.shape[1])
            self.index.add(feats)
        else:
            if NearestNeighbors is None:
                raise ImportError("Neither faiss nor scikit-learn is available")
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn.fit(feats)

    def _search(self, query_features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.features is None or self.proteins is None:
            raise RuntimeError("Call fit() before retrieve().")
        if k < 1 or k > len(self.features):
            raise ValueError(f"k must be in [1, {len(self.features)}], got {k}")
        query = l2_normalize_np(query_features)
        if self.use_faiss:
            scores, indices = self.index.search(query, k)
        else:
            distances, indices = self.nn.kneighbors(query, n_neighbors=k)
            scores = 1.0 - distances
        return indices.astype(np.int64), scores.astype(np.float32)

    def retrieve(
        self,
        query_features: np.ndarray,
        k: int,
        query_bank_indices: Optional[np.ndarray] = None,
        adaptive_temperature: bool = False,
        lambda_: float = 1.0,
        uniform: bool = False,
    ) -> FlowRetrievalResult:
        """Retrieve top-k neighbors and synthesize source protein/RNA states.

        query_bank_indices should contain the bank row corresponding to each
        query when queries are themselves bank members (training retrieval).
        Passing it removes the exact self row robustly instead of assuming that
        the first returned neighbor is always self.

        If adaptive_temperature is True, each cell gets its own softmax
        temperature tau_i = temperature * exp(lambda_ * (density_i - mean)),
        where density_i is the mean similarity of that cell's neighbors. Dense
        regions (many similar neighbors) are smoothed more; sparse regions stay
        sharp.
        """
        if self.features is None or self.proteins is None:
            raise RuntimeError("Call fit() before retrieve().")

        n_query = len(query_features)
        if query_bank_indices is not None:
            self_idx = np.asarray(query_bank_indices, dtype=np.int64)
            if self_idx.shape != (n_query,):
                raise ValueError("query_bank_indices must have shape [n_query]")
            if np.any(self_idx < 0) or np.any(self_idx >= len(self.features)):
                raise ValueError(
                    "query_bank_indices may only be supplied for queries that are actual "
                    "members of the fitted memory bank"
                )
            raw_k = min(k + 1, len(self.features))
            raw_indices, raw_scores = self._search(query_features, raw_k)
            indices = np.empty((n_query, k), dtype=np.int64)
            scores = np.empty((n_query, k), dtype=np.float32)
            for row in range(n_query):
                keep = raw_indices[row] != self_idx[row]
                row_idx = raw_indices[row][keep][:k]
                row_score = raw_scores[row][keep][:k]
                if len(row_idx) < k:
                    raise RuntimeError("Could not collect k non-self neighbors")
                indices[row] = row_idx
                scores[row] = row_score
        else:
            indices, scores = self._search(query_features, k)

        if uniform:
            weights = np.full((n_query, k), 1.0 / k, dtype=np.float32)
        else:
            if adaptive_temperature:
                density = scores.mean(axis=1)  # [N]
                density_mean = float(density.mean())
                tau = self.temperature * np.exp(lambda_ * (density - density_mean))
                logits = scores / tau[:, None]
            else:
                logits = scores / self.temperature
            logits = logits - logits.max(axis=1, keepdims=True)
            weights = np.exp(logits).astype(np.float32)
            weights /= weights.sum(axis=1, keepdims=True) + 1e-8

        protein_neighbors = self.proteins[indices]
        feature_neighbors = self.features[indices]
        protein_prior = np.einsum("bk,bkp->bp", weights, protein_neighbors).astype(np.float32)
        reference_feature = np.einsum("bk,bkd->bd", weights, feature_neighbors).astype(np.float32)
        reference_feature = l2_normalize_np(reference_feature)
        mean_similarity = np.sum(weights * scores, axis=1, keepdims=True).astype(np.float32)

        return FlowRetrievalResult(
            indices=indices,
            scores=scores,
            weights=weights,
            protein_prior=protein_prior,
            reference_feature=reference_feature,
            mean_similarity=mean_similarity,
        )
