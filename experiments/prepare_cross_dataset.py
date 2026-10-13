#!/usr/bin/env python3
"""Prepare explicit lymph-node reference / tonsil query input without fitting on query labels.

Requires equal gene and protein sets: names are reordered, never silently
intersected. Protein measurements from the query are retained for scoring only.
No alias matching, batch correction or target-derived feature selection is done.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_modules.protocol import split_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_h5ad", required=True)
    p.add_argument("--query_h5ad", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_seed", type=int, default=42)
    a = p.parse_args()
    import anndata as ad
    import numpy as np
    from metrics import ids, write_json
    ref, query = ad.read_h5ad(a.reference_h5ad), ad.read_h5ad(a.query_h5ad)
    genes = ids(ref.var_names, ref.n_vars, "reference genes")
    qgenes = ids(query.var_names, query.n_vars, "query genes")
    if set(genes) != set(qgenes):
        raise ValueError("Gene sets differ. Supply an explicitly aligned dataset; no silent intersection allowed")
    query = query[:, genes].copy()
    def protein_names(x):
        key = "protein_names" if "protein_names" in x.uns else "protein_var_names"
        return ids(x.uns[key], np.asarray(x.obsm["protein"]).shape[1], "proteins")
    names, qnames = protein_names(ref), protein_names(query)
    if set(names) != set(qnames):
        raise ValueError("Protein panels differ. Supply explicit biological mapping; channel count is insufficient")
    query.obsm["protein"] = np.asarray(query.obsm["protein"])[:, [qnames.index(n) for n in names]]
    original_ids = {}
    for role, data in (("reference", ref), ("query", query)):
        old = ids(data.obs_names, data.n_obs, role + " IDs")
        data.obs_names = [f"{role}::{x}" for x in old]
        original_ids.update(zip(data.obs_names, old))
        natural = data.obs["slice_id"].astype(str) if "slice_id" in data.obs else (
            data.obs["slice"].astype(str) if "slice" in data.obs else np.full(data.n_obs, "section"))
        data.obs["slice_id"] = [f"{role}::{x}" for x in natural]
        if "spatial" not in data.obsm:
            raise ValueError("Both datasets must supply spatial coordinates")
    # Concatenate only source RNA, target protein and coordinates needed by the model.
    combined = ad.concat([ref, query], join="inner", merge="same", uns_merge=None)
    combined.uns["protein_names"] = np.asarray(names, dtype=str)
    combined.uns["slice_roles"] = {x:("train" if x.startswith("reference::") else "test")
                                    for x in combined.obs["slice_id"].astype(str)}
    fit, val = split_reference(np.arange(ref.n_obs), split_seed=a.split_seed)
    te = np.arange(ref.n_obs, combined.n_obs)
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new empty experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out / "lymph_to_tonsil.h5ad")
    write_json(out / "split_manifest.json", {
        "fit_cell_ids": combined.obs_names[fit].tolist(),
        "validation_cell_ids": combined.obs_names[val].tolist(),
        "test_cell_ids": combined.obs_names[te].tolist()})
    write_json(out / "cross_dataset_provenance.json", {
        "reference":str(Path(a.reference_h5ad).resolve()), "query":str(Path(a.query_h5ad).resolve()),
        "reference_n":ref.n_obs, "query_n":query.n_obs, "split_seed":a.split_seed,
        "gene_names":genes, "protein_names":names, "original_cell_ids":original_ids,
        "protocol":"all supplied reference observations; all supplied query observations; strict named panels"})


if __name__ == "__main__":
    main()
