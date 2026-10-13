"""Build and cache Stage-1 (niche-fused GRN) + Stage-2 (retrieval) outputs.

Saves every array Stage-3 needs so that flow-model optimization can iterate
without re-running the first two stages. Uses the tuned config:
  niche fusion hidden=256, tau_spatial=2.0, alpha=20, k=150, retr_temp=0.35.

Example:
  python -u experiments/cache_stage12.py \
      --dataset lymph --data_path ../predata_standardized/Human_lymph_nodes.h5ad \
      --output_dir ../outputs/lymph_stage12_cache --device cuda
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
from core_modules.protocol import validate_three_way_split, split_reference
from core_modules.evaluation import context_for_fold, declared_split_indices
from core_modules.spatial_graph import (
    build_spatial_graph,
    compute_niche_context,
    gather_neighbor_features,
    make_slice_units,
)
from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from pipeline.config import get_spatial_config
from utils import log_normalize, select_per_protein_hvg_arrays, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["breast", "lymph", "postnatal", "postnatal_p21", "tonsil", "example"], required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42, help="Fixed partition seed, independent of training seed")
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--n_per_protein", type=int, default=100)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage1_lr", type=float, default=4e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--spatial_hidden", type=int, default=256)
    p.add_argument("--spatial_temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--use_niche", type=int, default=1, choices=[0, 1],
                   help="1=additive niche branch in the RNA encoder; 0=center RNA only")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--min_cells", type=int, default=0, help="Pre-filter genes: keep only genes expressed in >= min_cells cells (0=disabled)")
    p.add_argument("--k", type=int, default=150)
    p.add_argument("--retrieval_temperature", type=float, default=0.35)
    p.add_argument("--split_manifest", type=str, default=None, help="Shared fitting/validation/test cell-ID JSON; overrides automatic partitioning")
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


def train_stage1(model, arrays, lr, epochs, batch_size, device, alpha, beta, gamma):
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
        print(f"Stage1 epoch={epoch:03d} loss={running / max(batches, 1):.5f}", flush=True)
    model.eval()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scfg = get_spatial_config(args.dataset)
    scfg.spatial_temperature = args.spatial_temperature
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.data_path)

    shared = None
    if args.split_manifest:
        from types import SimpleNamespace
        from copy import deepcopy
        shared = declared_split_indices(args.split_manifest, adata.obs_names)
        roles = np.full(adata.n_obs, "reference", dtype=object)
        roles[shared[2]] = "query"
        obs = adata.obs.copy()
        natural = obs["slice_id"].astype(str).to_numpy() if "slice_id" in obs else (obs["slice"].astype(str).to_numpy() if "slice" in obs else np.full(adata.n_obs,"section"))
        units = np.asarray([f"{n}::{role}" for n,role in zip(natural,roles)])
        obs["slice_id"] = units
        graph_data = SimpleNamespace(obs=obs, obsm=adata.obsm, n_obs=adata.n_obs,
            uns={"slice_roles": {u:("test" if u.endswith("::query") else "train") for u in units}})
        graph_config=deepcopy(scfg);graph_config.split_mode="cross_slice"
        graph=build_spatial_graph(graph_data,graph_config)
        folds=[(np.concatenate(shared[:2]),shared[2])]
    else:
        graph = build_spatial_graph(adata, scfg)
        _, folds = make_slice_units(adata, scfg)

    X_full = adata.X
    var_names_full = list(adata.var_names)
    var_to_idx = {g: i for i, g in enumerate(var_names_full)}
    protein_raw = np.asarray(adata.obsm["protein"], dtype=np.float32)
    protein_norm = log_normalize(np.clip(protein_raw, 0.0, None))
    pn_key = "protein_names" if "protein_names" in adata.uns else "protein_var_names"
    protein_names = list(adata.uns[pn_key])

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if args.only_fold >= 0 and fold_id != args.only_fold:
            continue
        tr_idx, val_idx = split_reference(train_idx, split_seed=args.split_seed, fold=fold_id)
        if shared is not None:
            tr_idx, val_idx, test_idx = shared
        validate_three_way_split(tr_idx, val_idx, test_idx, n_obs=adata.n_obs)
        print(f"\n=== fold {fold_id}: train={len(tr_idx)} val={len(val_idx)} test={len(test_idx)} ===", flush=True)
        set_seed(args.seed + fold_id)

        X_tr_dense = X_full[tr_idx]
        if hasattr(X_tr_dense, "toarray"):
            X_tr_dense = X_tr_dense.toarray()
        X_tr_dense = X_tr_dense.astype(np.float32)

        # Any optional expression-frequency prefilter is estimated from fitting
        # cells only, so query/test RNA cannot influence gene selection.
        if args.min_cells > 0:
            eligible = (X_tr_dense > 0).sum(axis=0) >= args.min_cells
            X_select = X_tr_dense[:, eligible]
            names_select = [g for g, keep in zip(var_names_full, eligible) if keep]
        else:
            X_select = X_tr_dense
            names_select = var_names_full
        gene_names = select_per_protein_hvg_arrays(
            X_select, protein_raw[tr_idx], names_select,
            n_per_protein=args.n_per_protein, cap=args.n_top_genes,
        )
        gene_idx = np.array([var_to_idx[g] for g in gene_names], dtype=np.int64)

        rna_raw = X_full[:, gene_idx]
        if hasattr(rna_raw, "toarray"):
            rna_raw = rna_raw.toarray()
        rna_raw = rna_raw.astype(np.float32)
        rna_norm = log_normalize(rna_raw)

        # Split-aware spatial context. Training cells never consume validation or
        # test/query RNA as neighbors. Validation is evaluated as an observed
        # development query cohort; test/query context uses only test/query RNA.
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
            use_niche=bool(args.use_niche),
        ).to(device)
        train_stage1(stage1, (
            rna_norm[tr_idx], niche_rna[tr_idx], disp_ctx[tr_idx],
            rna_raw[tr_idx], protein_norm[tr_idx], protein_raw[tr_idx],
        ), args.stage1_lr, args.stage1_epochs, args.batch_size, device, args.alpha, args.beta, args.gamma)

        stage1.eval()
        z_all = l2_normalize_np(encode_rna_np(stage1, rna_norm, niche_rna, disp_ctx, device))
        neighbor_z = gather_neighbor_features(z_all, graph)
        neighbor_z = neighbor_z * split_valid[..., None].astype(np.float32)
        disp = graph.displacement.astype(np.float32).copy()
        disp[~split_valid] = 0.0

        bank = FlowMemoryBank(temperature=args.retrieval_temperature)
        bank_z = z_all[tr_idx]
        bank_pro = protein_norm[tr_idx]
        bank.fit(bank_z, bank_pro)
        n_tr = len(tr_idx)
        ret_tr = bank.retrieve(z_all[tr_idx], k=args.k, query_bank_indices=np.arange(n_tr))
        ret_va = bank.retrieve(z_all[val_idx], k=args.k)
        ret_te = bank.retrieve(z_all[test_idx], k=args.k)

        def sim1(x):
            return x if x.ndim == 2 else x[:, None]

        fold_dir = out / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if (fold_dir / "evaluation_context.json").exists():
            raise FileExistsError(f"{fold_dir}: immutable cache already exists; choose a new output directory")
        np.save(fold_dir / "cell_names.npy", np.asarray(adata.obs_names, dtype=str))
        np.save(fold_dir / "gene_names.npy", np.asarray(gene_names,dtype=str))
        np.save(fold_dir / "z_all.npy", z_all)
        np.save(fold_dir / "neighbor_z.npy", neighbor_z)
        np.save(fold_dir / "displacement.npy", disp)
        np.save(fold_dir / "rna_norm.npy", rna_norm)
        np.save(fold_dir / "niche_rna.npy", niche_rna)
        np.save(fold_dir / "disp_ctx.npy", disp_ctx)
        np.save(fold_dir / "protein_norm.npy", protein_norm)
        np.save(fold_dir / "protein_raw.npy", protein_raw)
        np.save(fold_dir / "protein_names.npy", np.asarray(protein_names))
        np.save(fold_dir / "train_idx.npy", tr_idx)
        np.save(fold_dir / "val_idx.npy", val_idx)
        np.save(fold_dir / "test_idx.npy", test_idx)
        np.save(fold_dir / "x0_train.npy", ret_tr.protein_prior)
        np.save(fold_dir / "x0_val.npy", ret_va.protein_prior)
        np.save(fold_dir / "x0_test.npy", ret_te.protein_prior)
        np.save(fold_dir / "sim_train.npy", sim1(ret_tr.mean_similarity))
        np.save(fold_dir / "sim_val.npy", sim1(ret_va.mean_similarity))
        np.save(fold_dir / "sim_test.npy", sim1(ret_te.mean_similarity))
        torch.save(stage1.state_dict(), fold_dir / "stage1_state.pt")
        from metrics import write_json
        write_json(fold_dir / "cache_config.json", vars(args))
        context_for_fold(fold_dir)

    print(f"\nDONE. cached to {out}")


if __name__ == "__main__":
    main()
