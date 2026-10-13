"""Data loading and utility helpers for scRAG-Flow."""

from __future__ import annotations

import json
import os
import random
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_cache_key(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def ensure_dense(x) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def log_normalize(data: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    row_sums = np.sum(data, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8
    return np.log1p(data / row_sums * target_sum).astype(np.float32)


def zscore_per_group(data: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Standardize each column within each group independently.

    Removes a per-slice batch effect so that train/test slices share a common
    scale (cross-slice domain alignment). Applied to the source (RNA) input only;
    the target protein stays in log-normalized count space.
    """
    out = np.asarray(data, dtype=np.float32).copy()
    for g in np.unique(group_ids):
        m = group_ids == g
        mean = out[m].mean(axis=0, keepdims=True)
        std = out[m].std(axis=0, keepdims=True) + 1e-8
        out[m] = (out[m] - mean) / std
    return out


def _to_numpy_frame(matrix) -> tuple[np.ndarray, list[str]]:
    if isinstance(matrix, pd.DataFrame):
        return matrix.values.astype(np.float32), list(matrix.columns)
    arr = np.asarray(matrix, dtype=np.float32)
    return arr, [f"protein_{i}" for i in range(arr.shape[1])]


def prepare_protein_matrices(adata: anndata.AnnData) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "protein" in adata.obsm:
        key = "protein"
    elif "protein_expression" in adata.obsm:
        key = "protein_expression"
    else:
        raise ValueError("Missing protein matrix in adata.obsm (expected 'protein' or 'protein_expression').")
    prot_raw, frame_names = _to_numpy_frame(adata.obsm[key])
    protein_names = list(adata.uns["protein_names"]) if "protein_names" in adata.uns else frame_names
    prot_norm = log_normalize(np.clip(prot_raw, a_min=0.0, a_max=None))
    return prot_raw.astype(np.float32), prot_norm.astype(np.float32), protein_names


@dataclass
class DataBundle:
    rna_norm_train: np.ndarray
    rna_norm_test: np.ndarray
    rna_raw_train: np.ndarray
    rna_raw_test: np.ndarray
    protein_norm_train: np.ndarray
    protein_norm_test: np.ndarray
    protein_raw_train: np.ndarray
    protein_raw_test: np.ndarray
    gene_names: list[str]
    protein_names: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray

    def meta(self) -> dict:
        payload = asdict(self)
        for key in list(payload):
            if isinstance(payload[key], np.ndarray):
                if key.endswith("_idx"):
                    payload[key] = payload[key].tolist()
                else:
                    payload.pop(key)
        payload["n_genes"] = len(self.gene_names)
        payload["n_proteins"] = len(self.protein_names)
        payload["n_train"] = len(self.train_idx)
        payload["n_test"] = len(self.test_idx)
        return payload


class MultiomeDataset(Dataset):
    def __init__(
        self,
        rna_norm: np.ndarray,
        rna_raw: np.ndarray,
        protein_norm: np.ndarray,
        protein_raw: np.ndarray,
    ) -> None:
        self.rna_norm = torch.from_numpy(rna_norm).float()
        self.rna_raw = torch.from_numpy(rna_raw).float()
        self.protein_norm = torch.from_numpy(protein_norm).float()
        self.protein_raw = torch.from_numpy(protein_raw).float()

    def __len__(self) -> int:
        return self.rna_norm.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "rna_norm": self.rna_norm[idx],
            "rna_raw": self.rna_raw[idx],
            "protein_norm": self.protein_norm[idx],
            "protein_raw": self.protein_raw[idx],
        }


def load_h5ad(path: str | Path) -> anndata.AnnData:
    return sc.read_h5ad(str(path))


def split_indices(n_obs: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_obs)
    n_train = int(n_obs * train_ratio)
    return indices[:n_train], indices[n_train:]


def select_protein_guided_genes(
    train_adata,
    protein: np.ndarray,
    n_top_genes: int = 1500,
) -> list[str]:
    """Select top genes by max absolute Pearson correlation with any protein."""
    import scipy.sparse as sp
    X = train_adata.X
    if sp.issparse(X):
        X = X.toarray()
    rna = X.astype(np.float64)
    prot = protein.astype(np.float64)
    n_genes = rna.shape[1]

    # Vectorized correlation: for each gene, max |r| across all proteins
    rna_c = rna - rna.mean(axis=0, keepdims=True)
    prot_c = prot - prot.mean(axis=0, keepdims=True)
    cov = rna_c.T @ prot_c
    std_rna = np.sqrt((rna_c ** 2).sum(axis=0)) + 1e-8
    std_prot = np.sqrt((prot_c ** 2).sum(axis=0)) + 1e-8
    corr = cov / np.outer(std_rna, std_prot)
    max_corr = np.max(np.abs(corr), axis=1)

    top_idx = np.argsort(max_corr)[-n_top_genes:]
    return list(np.array(train_adata.var_names)[top_idx])


def select_per_protein_hvg(
    train_adata,
    protein: np.ndarray,
    n_per_protein: int = 50,
) -> list[str]:
    """Fair HVG: each protein gets n_per_protein top-correlated genes; take union."""
    import scipy.sparse as sp
    X = train_adata.X
    if sp.issparse(X):
        X = X.toarray()
    rna = X.astype(np.float64)
    prot = protein.astype(np.float64)
    n_proteins = prot.shape[1]

    rna_c = rna - rna.mean(axis=0, keepdims=True)
    prot_c = prot - prot.mean(axis=0, keepdims=True)
    cov = rna_c.T @ prot_c
    std_rna = np.sqrt((rna_c ** 2).sum(axis=0)) + 1e-8
    std_prot = np.sqrt((prot_c ** 2).sum(axis=0)) + 1e-8
    corr = np.abs(cov / np.outer(std_rna, std_prot))  # [genes, proteins]

    gene_indices = set()
    for p in range(n_proteins):
        top_k = np.argsort(corr[:, p])[-n_per_protein:]
        gene_indices.update(top_k.tolist())

    gene_list = sorted(gene_indices)[:3000]  # cap at 3000
    return list(np.array(train_adata.var_names)[gene_list])


def select_per_protein_hvg_arrays(
    rna_raw: np.ndarray,
    protein_raw: np.ndarray,
    var_names: list[str],
    n_per_protein: int = 100,
    cap: int = 3000,
) -> list[str]:
    """Array-based per-protein HVG.

    For each protein, keep the top ``n_per_protein`` genes by absolute Pearson
    correlation, then take the union capped at ``cap`` genes. ``rna_raw`` and
    ``protein_raw`` must be aligned on the same cells (typically the train split
    only, to avoid feature-selection leakage into test protein labels).
    """
    rna = np.asarray(rna_raw, dtype=np.float64)
    prot = np.asarray(protein_raw, dtype=np.float64)
    rna_c = rna - rna.mean(axis=0, keepdims=True)
    prot_c = prot - prot.mean(axis=0, keepdims=True)
    cov = rna_c.T @ prot_c
    std_rna = np.sqrt((rna_c ** 2).sum(axis=0)) + 1e-8
    std_prot = np.sqrt((prot_c ** 2).sum(axis=0)) + 1e-8
    corr = np.abs(cov / np.outer(std_rna, std_prot))  # [genes, proteins]

    gene_indices = set()
    for p in range(prot.shape[1]):
        top_k = np.argsort(corr[:, p])[-n_per_protein:]
        gene_indices.update(top_k.tolist())

    # If the union exceeds the cap, retain the genes with the strongest
    # protein association rather than truncating by column/index order.
    gene_list = np.asarray(sorted(gene_indices), dtype=np.int64)
    if len(gene_list) > cap:
        score = corr[gene_list].max(axis=1)
        gene_list = gene_list[np.argsort(score)[-cap:]]
    return [var_names[int(i)] for i in gene_list]


def preprocess_single_dataset(
    data_path: str | Path,
    n_top_genes: int = 1000,
    train_ratio: float = 0.8,
    seed: int = 42,
    use_protein_guided_hvg: bool = False,
    use_per_protein_hvg: bool = False,
    n_per_protein: int = 50,
    fold_train_idx: np.ndarray | None = None,
    fold_test_idx: np.ndarray | None = None,
    gene_names: list[str] | None = None,  # skip HVG if provided
) -> DataBundle:
    adata = load_h5ad(data_path)
    protein_raw, protein_norm, protein_names = prepare_protein_matrices(adata)
    if fold_train_idx is not None and fold_test_idx is not None:
        train_idx = fold_train_idx
        test_idx = fold_test_idx
    else:
        train_idx, test_idx = split_indices(adata.n_obs, train_ratio, seed)

    if gene_names is not None:
        hvg_names = [g for g in gene_names if g in adata.var_names]
    else:
        train_adata = adata[train_idx].copy()
        if use_per_protein_hvg:
            hvg_names = select_per_protein_hvg(train_adata, protein_raw[train_idx], n_per_protein)
        elif use_protein_guided_hvg:
            hvg_names = select_protein_guided_genes(train_adata, protein_raw[train_idx], n_top_genes)
        else:
            sc.pp.normalize_total(train_adata, target_sum=1e4)
            sc.pp.log1p(train_adata)
            sc.pp.highly_variable_genes(train_adata, n_top_genes=n_top_genes)
            hvg_names = train_adata.var_names[train_adata.var["highly_variable"]].tolist()

    adata_hvg = adata[:, hvg_names].copy()
    rna_raw = ensure_dense(adata_hvg.X).astype(np.float32)
    rna_norm = log_normalize(rna_raw)
    return DataBundle(
        rna_norm_train=rna_norm[train_idx],
        rna_norm_test=rna_norm[test_idx],
        rna_raw_train=rna_raw[train_idx],
        rna_raw_test=rna_raw[test_idx],
        protein_norm_train=protein_norm[train_idx],
        protein_norm_test=protein_norm[test_idx],
        protein_raw_train=protein_raw[train_idx],
        protein_raw_test=protein_raw[test_idx],
        gene_names=hvg_names,
        protein_names=protein_names,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def preprocess_cross_dataset(
    train_path: str | Path,
    test_path: str | Path,
    n_top_genes: int = 1000,
) -> DataBundle:
    train_adata = load_h5ad(train_path)
    test_adata = load_h5ad(test_path)
    train_protein_raw, train_protein_norm, protein_names = prepare_protein_matrices(train_adata)
    test_protein_raw, test_protein_norm, _ = prepare_protein_matrices(test_adata)

    hvg_adata = train_adata.copy()
    sc.pp.normalize_total(hvg_adata, target_sum=1e4)
    sc.pp.log1p(hvg_adata)
    sc.pp.highly_variable_genes(hvg_adata, n_top_genes=n_top_genes)
    hvg_names = [g for g in hvg_adata.var_names[hvg_adata.var["highly_variable"]] if g in test_adata.var_names]

    train_hvg = train_adata[:, hvg_names].copy()
    test_hvg = test_adata[:, hvg_names].copy()

    train_rna_raw = ensure_dense(train_hvg.X).astype(np.float32)
    test_rna_raw = ensure_dense(test_hvg.X).astype(np.float32)

    return DataBundle(
        rna_norm_train=log_normalize(train_rna_raw),
        rna_norm_test=log_normalize(test_rna_raw),
        rna_raw_train=train_rna_raw,
        rna_raw_test=test_rna_raw,
        protein_norm_train=train_protein_norm.astype(np.float32),
        protein_norm_test=test_protein_norm.astype(np.float32),
        protein_raw_train=train_protein_raw.astype(np.float32),
        protein_raw_test=test_protein_raw.astype(np.float32),
        gene_names=list(hvg_names),
        protein_names=protein_names,
        train_idx=np.arange(train_adata.n_obs),
        test_idx=np.arange(test_adata.n_obs),
    )


def create_dataloaders(
    bundle: DataBundle,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MultiomeDataset(
        bundle.rna_norm_train,
        bundle.rna_raw_train,
        bundle.protein_norm_train,
        bundle.protein_raw_train,
    )
    test_ds = MultiomeDataset(
        bundle.rna_norm_test,
        bundle.rna_raw_test,
        bundle.protein_norm_test,
        bundle.protein_raw_test,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_bundle_meta(path: str | Path, bundle: DataBundle) -> None:
    save_json(path, bundle.meta())
