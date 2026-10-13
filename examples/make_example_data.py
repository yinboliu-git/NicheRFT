#!/usr/bin/env python3
"""Generate a small, fully synthetic example dataset for the quick-start demo.

The file is created on the fly so no real tissue data (and no identifying
information) ships with the repository. It reproduces the input schema that the
pipeline expects:

  adata.X                      RNA counts, cells x genes (raw, non-negative)
  adata.obsm["protein"]        raw protein (ADT) abundance, cells x P
  adata.obsm["spatial"]        (x, y) per cell; each slice has its own frame
  adata.obs["slice_id"]        "slice1" / "slice2"
  adata.obs["split"]           "train" / "test"
  adata.uns["protein_names"]   protein panel
  adata.uns["slice_roles"]     {"slice1": "train", "slice2": "test"}

Two spatially gridded slices share a latent program; both RNA and protein are
generated from that program, so a model that recovers it can transfer the
retrieval prior across slices. slice2 carries a small domain shift, mirroring
the cross-slice task.

Example:
  python examples/make_example_data.py --output_file examples/data/example.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import anndata as ad


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _latent(coords: np.ndarray, phase: float, rng: np.random.Generator, n_factors: int) -> np.ndarray:
    """Smooth spatial latent program; neighbours share factors."""
    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    z = np.zeros((len(coords), n_factors), dtype=np.float64)
    for k in range(n_factors):
        z[:, k] = (
            1.1 * np.sin((0.35 + 0.12 * k) * x + phase + k)
            + 0.9 * np.cos((0.45 + 0.08 * k) * y - phase)
            + 0.35 * rng.standard_normal(len(coords))
        )
    return z


def _grid(n_cols: int, n_rows: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


def build_slice(
    coords: np.ndarray,
    phase: float,
    gene_load: np.ndarray,
    gene_bias: np.ndarray,
    prot_load: np.ndarray,
    prot_bias: np.ndarray,
    rng: np.random.Generator,
    n_factors: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    z = _latent(coords, phase, rng, n_factors)
    gene_rate = _softplus(gene_bias[None, :] + z @ gene_load.T) * 3.0
    prot_rate = _softplus(prot_bias[None, :] + z @ prot_load.T) * 4.0
    rna = rng.poisson(gene_rate).astype(np.float32)
    prot = rng.poisson(prot_rate).astype(np.float32)
    return sp.csr_matrix(rna), prot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_genes", type=int, default=2000)
    p.add_argument("--n_proteins", type=int, default=16)
    p.add_argument("--n_factors", type=int, default=6)
    p.add_argument("--slice1_cols", type=int, default=24)
    p.add_argument("--slice1_rows", type=int, default=16)
    p.add_argument("--slice2_cols", type=int, default=24)
    p.add_argument("--slice2_rows", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    K, G, P = args.n_factors, args.n_genes, args.n_proteins

    # Shared generative program: sparse gene loadings, dense-ish protein loadings.
    gene_load = np.zeros((G, K), dtype=np.float64)
    for g in range(G):
        for k in rng.choice(K, size=2, replace=False):
            gene_load[g, k] = rng.normal(0.0, 0.7)
    gene_bias = rng.normal(-1.0, 0.4, size=G)
    prot_load = rng.normal(0.0, 0.8, size=(P, K))
    prot_bias = rng.normal(0.2, 0.3, size=P)

    coords1 = _grid(args.slice1_cols, args.slice1_rows)
    coords2 = _grid(args.slice2_cols, args.slice2_rows) + np.array([0.0, args.slice1_rows + 4.0])
    x1, p1 = build_slice(coords1, 0.0, gene_load, gene_bias, prot_load, prot_bias, rng, K)
    x2, p2 = build_slice(coords2, 0.8, gene_load, gene_bias, prot_load, prot_bias, rng, K)

    rna = sp.vstack([x1, x2]).tocsr()
    protein = np.vstack([p1, p2]).astype(np.float32)
    spatial = np.vstack([coords1, coords2]).astype(np.float32)

    n1, n2 = len(coords1), len(coords2)
    gene_names = [f"G{g:04d}" for g in range(G)]
    protein_names = [f"P{p:02d}" for p in range(P)]
    cell_names = [f"C{i:06d}" for i in range(n1 + n2)]
    slice_id = np.array(["slice1"] * n1 + ["slice2"] * n2)
    split = np.array(["train"] * n1 + ["test"] * n2)

    adata = ad.AnnData(X=rna)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.obsm["protein"] = protein
    adata.obsm["spatial"] = spatial
    adata.obs["slice_id"] = slice_id
    adata.obs["split"] = split
    adata.obs["x"] = spatial[:, 0]
    adata.obs["y"] = spatial[:, 1]
    adata.uns["protein_names"] = protein_names
    adata.uns["has_protein"] = True
    adata.uns["has_spatial"] = True
    adata.uns["slice_roles"] = {"slice1": "train", "slice2": "test"}
    adata.uns["task"] = "RNA_to_protein"
    adata.uns["source_modality"] = "RNA"
    adata.uns["target_modality"] = "protein"
    adata.uns["schema_version"] = "rna_protein_spatial_v1"
    adata.uns["preprocessing"] = {
        "note": "synthetic example; generated by examples/make_example_data.py",
        "normalization": "not performed",
    }

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"wrote {out}  shape={adata.shape}  "
          f"slice1={n1} slice2={n2}  proteins={P}", flush=True)


if __name__ == "__main__":
    main()
