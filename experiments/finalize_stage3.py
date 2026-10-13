"""Finalize Stage-3 (CAP-Flow) for a fixed config and persist all artifacts.

Reads a fixed Stage-I/II cache, trains pure conditional flow matching, selects
by validation PCC-P, and writes named predictions under the canonical evaluator.
The evaluation source bandwidth is fitted on fitting proteins only.

Example:
  python -u experiments/finalize_stage3.py \
      --cache_dir ../outputs/lymph_stage12_cache \
      --output_dir ../outputs/lymph_stage3_final --seed 42
"""
from __future__ import annotations

import argparse
import json
import hashlib
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from core_modules.flow_retrieval import FlowMemoryBank
from core_modules.protocol import validate_three_way_split
from metrics import compute_pcc, write_json
from core_modules.evaluation import context_for_fold, score_array, save_predictions
from models.cap_flow import (CapFlow, cap_flow_matching_loss, integrate_cap_flow,
                             matched_residual_loss, matched_residual_prediction)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", choices=["flow", "residual"], default="flow",
                   help="Matched DiT residual control uses direct supervised prediction, no ODE")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--k", type=int, default=150)
    p.add_argument("--retrieval_temperature", type=float, default=0.35)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--per_protein_head", action="store_true")
    p.add_argument("--sigma_clip", type=float, default=1e-3)
    p.add_argument("--de_shrink_strength", type=float, default=1.0)
    p.add_argument("--flow_path", type=str, default="straight", choices=["straight", "trig", "noise"])
    p.add_argument("--no_spatial_tokens", action="store_true",
                   help="Zero out neighbor_z and displacement so the flow carries no spatial-niche signal.")
    p.add_argument("--uniform_retrieval", action="store_true",
                   help="Uniform (1/k) neighbor aggregation instead of similarity-weighted.")
    p.add_argument("--prediction_nfe", type=int, default=8, help="Prespecified deployment and validation step count")
    p.add_argument("--nfe", type=int, nargs="+", default=[8, 16])
    p.add_argument("--slice_relative_center", action="store_true",
                   help="Center the de-shrink calibration on each slice's own retrieval-prior "
                        "mean (label-free, transductive). Training already uses the fitting-slice "
                        "mean; this applies the same slice-relative rule at validation and test.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _to_loader(*arrays, batch_size, shuffle):
    t = [torch.from_numpy(np.asarray(a, np.float32)) for a in arrays]
    return DataLoader(TensorDataset(*t), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_cache(cache_dir: Path, fold: int = 0):
    fold = cache_dir / f"fold{fold}"
    context_for_fold(fold)
    return {
        "z_all": np.load(fold / "z_all.npy"),
        "neighbor_z": np.load(fold / "neighbor_z.npy"),
        "displacement": np.load(fold / "displacement.npy"),
        "protein_norm": np.load(fold / "protein_norm.npy"),
        "protein_names": list(np.load(fold / "protein_names.npy", allow_pickle=True)),
        "train_idx": np.load(fold / "train_idx.npy"),
        "val_idx": np.load(fold / "val_idx.npy"),
        "test_idx": np.load(fold / "test_idx.npy"),
    }


def retrieve_context(data, k, temp, uniform=False):
    z = data["z_all"]; pn = data["protein_norm"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    validate_three_way_split(tr, va, te, n_obs=len(z))
    bank = FlowMemoryBank(temperature=temp)
    # Strict split-aware retrieval: the memory bank contains fitting cells only.
    # Validation/test cells are queries only and are never inserted into the bank.
    bank_z = z[tr]
    bank_pro = pn[tr]
    bank.fit(bank_z, bank_pro)
    n_tr = len(tr)
    r_tr = bank.retrieve(z[tr], k=k, query_bank_indices=np.arange(n_tr), uniform=uniform)
    r_va = bank.retrieve(z[va], k=k, uniform=uniform)
    r_te = bank.retrieve(z[te], k=k, uniform=uniform)

    def gather(r):
        sim = r.mean_similarity if r.mean_similarity.ndim == 2 else r.mean_similarity[:, None]
        return r.protein_prior, r.reference_feature, bank_z[r.indices[:, :8]], sim

    return gather(r_tr), gather(r_va), gather(r_te)


@torch.no_grad()
def predict(model, x0, q, ref, retz, sz, disp, sim, device, nfe=8, stochastic=False, flow_path="straight", sigma_clip=1e-3, objective="flow"):
    model.eval()
    outs = []
    rng_devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    rng_guard = torch.random.fork_rng(devices=rng_devices)
    with rng_guard:
        torch.manual_seed(42)
        for i in range(0, len(x0), 1024):
            s = slice(i, i + 1024)
            inputs = [torch.from_numpy(a[s]).to(device) for a in (x0, q, ref, retz, sz, disp, sim)]
            if objective == "residual":
                pred = matched_residual_prediction(model, *inputs)
            else:
                pred = integrate_cap_flow(model, *inputs, nfe=nfe, stochastic=stochastic,
                                          flow_path=flow_path, sigma_clip=sigma_clip)
            outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _metrics_serializable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.objective == "residual" and args.flow_path != "straight":
        raise ValueError("The matched residual control requires the same calibrated source as the full model")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir) / f"fold{args.fold}" / f"seed{args.seed}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir}: existing run artifacts; use a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_cache(cache_dir, args.fold)
    z = data["z_all"]; nz = data["neighbor_z"]; disp = data["displacement"]
    pn = data["protein_norm"]; names = data["protein_names"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    validate_three_way_split(tr, va, te, n_obs=len(z))

    folder = cache_dir / f"fold{args.fold}"
    context = context_for_fold(folder)
    if args.epochs < 1 or args.prediction_nfe < 1 or any(n < 1 for n in args.nfe):
        raise ValueError("Epoch and NFE counts must be positive")
    def report(pred):
        return score_array(pred, folder, context=context)

    if args.no_spatial_tokens:
        nz = np.zeros_like(nz)
        disp = np.zeros_like(disp)

    (x0_tr, ref_tr, retz_tr, sim_tr), (x0_va, ref_va, retz_va, sim_va), (x0_te, ref_te, retz_te, sim_te) = \
        retrieve_context(data, args.k, args.retrieval_temperature, uniform=args.uniform_retrieval)

    n_protein = len(names); rna_dim = z.shape[1]; n_near = nz.shape[1] // 2

    model = CapFlow(
        protein_dim=n_protein, rna_dim=rna_dim, hidden_dim=512, depth=4, n_heads=8,
        dropout=0.1, n_near=n_near, k_retrieved=8, prior_hidden=128,
        rank=(args.rank if args.rank > 0 else None),
        per_protein_head=args.per_protein_head,
    ).to(device)
    # All source-distribution statistics are estimated from fitting cells only.
    model.prior_adapter.set_retrieval_mean(x0_tr.mean(axis=0))
    resid_var = np.clip(np.var(pn[tr] - x0_tr, axis=0), 1e-3, None)
    model.prior_adapter.set_logvar_bias(np.log(resid_var))
    shrink = np.clip(x0_tr.std(axis=0) / (pn[tr].std(axis=0) + 1e-8), 1e-3, None)
    de_shrink = 1.0 + args.de_shrink_strength * (1.0 / shrink - 1.0)
    model.prior_adapter.set_de_shrink(np.clip(de_shrink, 0.5, 4.0).astype(np.float32))

    loader = _to_loader(x0_tr, z[tr], ref_tr, retz_tr, nz[tr], disp[tr], sim_tr, pn[tr],
                        batch_size=1024, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_val = -float("inf"); best_state = deepcopy(model.state_dict())
    for ep in range(1, args.epochs + 1):
        model.train()
        for x0b, qb, rb, rzb, szb, dpb, sb, yb in loader:
            x0b = x0b.to(device); qb = qb.to(device); rb = rb.to(device); rzb = rzb.to(device)
            szb = szb.to(device); dpb = dpb.to(device); sb = sb.to(device); yb = yb.to(device)
            if args.objective == "residual":
                loss = matched_residual_loss(model, x0b, yb, qb, rb, rzb, szb, dpb, sb)
            else:
                loss = cap_flow_matching_loss(
                    model, x0b, yb, qb, rb, rzb, szb, dpb, sb,
                    sigma_clip=args.sigma_clip, flow_path=args.flow_path,
                ).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 2 == 0 or ep <= 2 or ep == args.epochs:
            if args.slice_relative_center:
                model.prior_adapter.set_retrieval_mean(x0_va.mean(axis=0))
            val_pred = predict(model, x0_va, z[va], ref_va, retz_va, nz[va], disp[va], sim_va, device, nfe=args.prediction_nfe, flow_path=args.flow_path, sigma_clip=args.sigma_clip, objective=args.objective)
            if args.slice_relative_center:
                model.prior_adapter.set_retrieval_mean(x0_tr.mean(axis=0))
            val_pcc = compute_pcc(val_pred, pn[va], names)[2]
            if val_pcc > best_val:
                best_val = val_pcc
                best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    # Slice-relative source center: de-shrink about the query slice's own
    # retrieval-prior mean (label-free, transductive), matching the relative-space
    # design. Fitting cells already use the fitting-slice mean during training.
    if args.slice_relative_center:
        model.prior_adapter.set_retrieval_mean(x0_te.mean(axis=0))

    # Test metrics are computed only after checkpoint selection is complete.
    ret_metrics = report(x0_te)
    flow_pred = predict(model, x0_te, z[te], ref_te, retz_te, nz[te], disp[te], sim_te, device, nfe=args.prediction_nfe, flow_path=args.flow_path, sigma_clip=args.sigma_clip, objective=args.objective)
    flow_stoch = None if args.objective == "residual" else predict(model, x0_te, z[te], ref_te, retz_te, nz[te], disp[te], sim_te, device, nfe=args.prediction_nfe, stochastic=True, flow_path=args.flow_path, sigma_clip=args.sigma_clip, objective=args.objective)
    flow_metrics = report(flow_pred)
    mu = model.prior_adapter.retrieval_mean.detach().cpu().numpy()
    kappa = model.prior_adapter.de_shrink.detach().cpu().numpy()
    calibrated_pred = mu + (x0_te - mu) * kappa
    calibrated_metrics = report(calibrated_pred)
    sweep = {}

    for nfe in (sorted(set(args.nfe)) if args.objective == "flow" else []):
        fp = predict(model, x0_te, z[te], ref_te, retz_te, nz[te], disp[te], sim_te, device, nfe=nfe, flow_path=args.flow_path, sigma_clip=args.sigma_clip, objective=args.objective)
        m = report(fp)
        sweep[str(nfe)] = _metrics_serializable(m)
        save_predictions(out_dir / f"flow_pred_test_nfe{nfe}.npz", fp, folder,
                         metadata={"nfe":nfe,"seed":args.seed,"flow_path":args.flow_path})
        print(f"[seed {args.seed}] NFE={nfe}: PCC-P={m['pcc_protein_mean']:.4f} "
              f"PCC-C={m['pcc_cell_mean']:.4f} MMD={m['mmd']:.4f}", flush=True)

    torch.save(best_state, out_dir / "flow_model.pt")
    np.save(out_dir / "retrieval_pred_test.npy", x0_te)
    np.save(out_dir / "flow_pred_test.npy", flow_pred)
    if flow_stoch is not None:
        np.save(out_dir / "flow_pred_test_stochastic.npy", flow_stoch)
    np.save(out_dir / "calibrated_prior_pred_test.npy", calibrated_pred)
    for label, values in [("retrieval",x0_te),("calibrated_prior",calibrated_pred),("flow",flow_pred),("flow_stochastic",flow_stoch)]:
        if values is None:
            continue
        save_predictions(out_dir / f"{label}_pred_test.npz", values, folder,
                         metadata={"kind":label,"config":vars(args)})
    cache_config_path = folder / "cache_config.json"
    cache_config = json.loads(cache_config_path.read_text()) if cache_config_path.exists() else None
    write_json(out_dir / "run_config.json", {"arguments":vars(args),
        "objective":args.objective,"stage12_config":cache_config,
        "cache_dir":str(cache_dir.resolve()),
        "stage1_checkpoint_hash":hashlib.sha256((folder / "stage1_state.pt").read_bytes()).hexdigest() if (folder / "stage1_state.pt").exists() else None,
        "context_fingerprint":context.fingerprint,"selection_metric":"validation PCC-P",
        "ode_nfe":args.prediction_nfe if args.objective == "flow" else 0,
        "predictor_calls":args.prediction_nfe if args.objective == "flow" else 1,
        "validation_nfe":args.prediction_nfe if args.objective == "flow" else 0,
        "source_statistics":("slice-relative retrieval-prior center at validation/test; "
                             "kappa and residual variance from fitting cells") if args.slice_relative_center
                             else "fixed fitting-reference center, kappa and residual variance",
        "model":{"protein_dim":n_protein,"rna_dim":rna_dim,"hidden_dim":512,"depth":4,
                 "n_heads":8,"n_near":n_near,"k_retrieved":8,"dropout":0.1,
                 "rank":args.rank if args.rank>0 else None,"per_protein_head":args.per_protein_head}})
    write_json(out_dir / "metrics.json", {
            "seed": args.seed,
            "config": {
                "k": args.k, "retrieval_temperature": args.retrieval_temperature,
                "prediction_nfe":args.prediction_nfe if args.objective == "flow" else 0,
                "objective":args.objective,"flow_path":args.flow_path,
                "slice_relative_center":args.slice_relative_center,
                "hidden_dim": 512, "depth": 4, "n_heads": 8, "epochs": args.epochs,
                "neighbor_mask_prob": 0, "mask_prob": 0, "cond_drop_prob": 0,
            },
            "retrieval": _metrics_serializable(ret_metrics),
            "calibrated_prior": _metrics_serializable(calibrated_metrics),
            "sweep":sweep,
            "evaluation_context":context.fingerprint,
            "flow": _metrics_serializable(flow_metrics),
        })

    def brief(m):
        return (f"PCC-P={m['pcc_protein_mean']:.4f} PCC-C={m['pcc_cell_mean']:.4f} "
                f"CMD_prot={m['cmd_protein']:.4f} CMD_cell={m['cmd_cell']:.4f} "
                f"RMSE={m['rmse']:.4f} MMD={m['mmd']:.4f}")
    print(f"[seed {args.seed}] retrieval: {brief(ret_metrics)}", flush=True)
    print(f"[seed {args.seed}] flow     : {brief(flow_metrics)}", flush=True)
    print(f"[seed {args.seed}] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
