"""Focused Stage-1 + Stage-2 (retrieval) evaluation for the niche-fused GRN.

Trains the niche-aware Stage-1 encoder and reports the FAISS retrieval PCC-P
without running Stage-3, so the spatial fusion can be evaluated in isolation.

Example:
  python -u experiments/test_stage1_retrieval.py \
      --dataset lymph --data_path ../predata_standardized/Human_lymph_nodes.h5ad \
      --stage1_epochs 20 --spatial_temperature 1.0 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from core_modules.flow_retrieval import FlowMemoryBank, l2_normalize_np
from core_modules.spatial_graph import (
    build_spatial_graph,
    compute_niche_context,
    make_slice_units,
)
from metrics import compute_pcc
from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from pipeline.config import get_spatial_config
from utils import log_normalize, save_json, select_per_protein_hvg_arrays, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["breast", "lymph", "postnatal"], required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--n_per_protein", type=int, default=100)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage1_lr", type=float, default=4e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--retrieval_temperature", type=float, default=0.35)
    p.add_argument("--spatial_temperature", type=float, default=1.0)
    p.add_argument("--spatial_hidden", type=int, default=128)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--adaptive_temperature", action="store_true")
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--only_fold", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def encode_rna_np(model, center, niche, disp, device, bs: int = 4096):
    outs = []
    for i in range(0, len(center), bs):
        c = torch.from_numpy(center[i:i + bs]).to(device)
        n = torch.from_numpy(niche[i:i + bs]).to(device)
        d = torch.from_numpy(disp[i:i + bs]).to(device)
        outs.append(model.encode_rna(c, n, d).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def train_stage1(model, arrays, lr, epochs, batch_size, device, log_fn, alpha=10.0, beta=0.5, gamma=0.5):
    model.train()
    weights = Stage1LossWeights(alpha, beta, gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rna_norm, niche_rna, disp_ctx, rna_raw, prot_norm, prot_raw = arrays
    ds = TensorDataset(
        torch.from_numpy(rna_norm), torch.from_numpy(niche_rna), torch.from_numpy(disp_ctx),
        torch.from_numpy(rna_raw), torch.from_numpy(prot_norm), torch.from_numpy(prot_raw),
    )
    loader = DataLoader(ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    for epoch in range(1, epochs + 1):
        running, batches = 0.0, 0
        for rn, nic, dc, rr, pn, pr in loader:
            batch = {
                "rna_norm": rn.to(device), "niche_rna": nic.to(device), "disp_ctx": dc.to(device),
                "rna_raw": rr.to(device), "protein_norm": pn.to(device), "protein_raw": pr.to(device),
            }
            out = model.compute_loss(batch, weights)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(out["loss"].detach())
            batches += 1
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            log_fn(f"Stage1 epoch={epoch:03d} loss={running / max(batches, 1):.5f}")
    model.eval()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scfg = get_spatial_config(args.dataset)
    if args.spatial_temperature is not None:
        scfg.spatial_temperature = args.spatial_temperature

    adata = sc.read_h5ad(args.data_path)
    graph = build_spatial_graph(adata, scfg)
    _, folds = make_slice_units(adata, scfg)

    X_full = adata.X
    var_names_full = list(adata.var_names)
    var_to_idx = {g: i for i, g in enumerate(var_names_full)}
    protein_raw = np.asarray(adata.obsm["protein"], dtype=np.float32)
    protein_norm = log_normalize(np.clip(protein_raw, 0.0, None))
    protein_names = list(adata.uns["protein_names"])

    log_fn = print
    results = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if args.only_fold >= 0 and fold_id != args.only_fold:
            continue
        rng = np.random.default_rng(args.split_seed + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(1, int(len(train_idx) * 0.125))
        val_idx = train_idx[perm[:n_val]]
        tr_idx = train_idx[perm[n_val:]]
        log_fn(f"\n=== fold {fold_id}: train={len(tr_idx)} val={len(val_idx)} test={len(test_idx)} ===")
        set_seed(args.seed + fold_id)

        X_tr_dense = X_full[tr_idx]
        if hasattr(X_tr_dense, "toarray"):
            X_tr_dense = X_tr_dense.toarray()
        X_tr_dense = np.asarray(X_tr_dense, dtype=np.float32)
        gene_names = select_per_protein_hvg_arrays(
            X_tr_dense, protein_raw[tr_idx], var_names_full,
            n_per_protein=args.n_per_protein, cap=args.n_top_genes,
        )
        gene_idx = np.array([var_to_idx[g] for g in gene_names], dtype=np.int64)
        log_fn(f"protein-guided feature selection retained {len(gene_names)} genes")

        rna_raw = X_full[:, gene_idx]
        if hasattr(rna_raw, "toarray"):
            rna_raw = rna_raw.toarray()
        rna_raw = np.asarray(rna_raw, dtype=np.float32)
        rna_norm = log_normalize(rna_raw)
        neighbor_idx = graph.neighbor_indices
        split_valid = np.zeros_like(graph.valid, dtype=bool)
        development_idx = np.concatenate([tr_idx, val_idx])
        split_valid[tr_idx] = graph.valid[tr_idx] & np.isin(neighbor_idx[tr_idx], tr_idx)
        split_valid[val_idx] = graph.valid[val_idx] & np.isin(neighbor_idx[val_idx], development_idx)
        split_valid[test_idx] = graph.valid[test_idx] & np.isin(neighbor_idx[test_idx], test_idx)
        niche_rna, disp_ctx = compute_niche_context(
            rna_norm, graph, spatial_temperature=scfg.spatial_temperature, valid_mask=split_valid
        )

        stage1 = ContrastiveAutoencoder(
            rna_dim=len(gene_names), protein_dim=len(protein_names),
            hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, proj_dim=args.proj_dim,
            depth=args.depth, dropout=args.dropout, spatial_hidden=args.spatial_hidden,
        ).to(device)
        train_stage1(stage1, (
            rna_norm[tr_idx], niche_rna[tr_idx], disp_ctx[tr_idx],
            rna_raw[tr_idx], protein_norm[tr_idx], protein_raw[tr_idx],
        ), args.stage1_lr, args.stage1_epochs, args.batch_size, device, log_fn,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma)

        z_all = l2_normalize_np(encode_rna_np(stage1, rna_norm, niche_rna, disp_ctx, device))
        bank = FlowMemoryBank(temperature=args.retrieval_temperature)
        bank_z = z_all[tr_idx]
        bank_pro = protein_norm[tr_idx]
        bank.fit(bank_z, bank_pro)
        n_tr = len(tr_idx)
        ret_te = bank.retrieve(
            z_all[test_idx], k=args.k,
            adaptive_temperature=args.adaptive_temperature, lambda_=args.lambda_,
        )

        m = {"pcc_protein_mean": compute_pcc(ret_te.protein_prior, protein_norm[test_idx], protein_names)[2]}
        log_fn(f"Retrieval PCC-P={m['pcc_protein_mean']:.4f}")
        results.append({"fold": fold_id, "retrieval_pcc_protein": float(m["pcc_protein_mean"])})

    if results:
        mean = float(np.mean([r["retrieval_pcc_protein"] for r in results]))
        print(f"\nDONE. mean retrieval PCC-P = {mean:.4f}")
        save_json(ROOT / "outputs" / f"{args.dataset}_retrieval.json", {"results": results, "mean": mean})


if __name__ == "__main__":
    main()
