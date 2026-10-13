"""Stage-3 CAP-Flow training/evaluation that reads the Stage-1/Stage-2 cache.

Loads the cached z_rna / neighbor_z / displacement / protein, re-computes the
retrieval (cheap FAISS) to obtain the retrieved reference RNA + retrieved-neighbor
tokens, then trains the conditional prior flow (CAP-Flow): a PriorAdapter
de-shrinks the retrieval prior into a Gaussian prior, and a DiT flow transports it
to the protein target.

Example:
  python -u experiments/run_stage3_capflow.py \
      --cache_dir ../outputs/lymph_stage12_cache --device cuda
"""
from __future__ import annotations

import argparse
import random
import time
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
from models.cap_flow import CapFlow, cap_flow_matching_loss, integrate_cap_flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--flow_epochs", type=int, default=20)
    p.add_argument("--flow_hidden_dim", type=int, default=512)
    p.add_argument("--flow_depth", type=int, default=4)
    p.add_argument("--flow_n_heads", type=int, default=8)
    p.add_argument("--flow_lr", type=float, default=3e-4)
    p.add_argument("--flow_dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--prior_hidden", type=int, default=128)
    p.add_argument("--sigma_clip", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--mask_prob", type=float, default=0.0)
    p.add_argument("--neighbor_mask_prob", type=float, default=0.0)
    p.add_argument("--cond_drop_prob", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, nargs="+", default=[1.0])
    p.add_argument("--k", type=int, default=150)
    p.add_argument("--k_retrieved", type=int, default=8)
    p.add_argument("--retrieval_temperature", type=float, default=0.35)
    p.add_argument("--nfe", type=int, nargs="+", default=[8, 16])
    p.add_argument("--eval_every", type=int, default=2)
    return p.parse_args()


def _to_loader(*arrays, batch_size, shuffle):
    tensors = [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_cache(cache_dir: Path):
    fold_dir = cache_dir / "fold0"
    data = {
        "z_all": np.load(fold_dir / "z_all.npy"),
        "neighbor_z": np.load(fold_dir / "neighbor_z.npy"),
        "displacement": np.load(fold_dir / "displacement.npy"),
        "protein_norm": np.load(fold_dir / "protein_norm.npy"),
        "protein_names": list(np.load(fold_dir / "protein_names.npy", allow_pickle=True)),
        "train_idx": np.load(fold_dir / "train_idx.npy"),
        "val_idx": np.load(fold_dir / "val_idx.npy"),
        "test_idx": np.load(fold_dir / "test_idx.npy"),
    }
    return data


def retrieve_context(data, k, k_retrieved, temperature):
    z_all = data["z_all"]
    protein_norm = data["protein_norm"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    validate_three_way_split(tr, va, te, n_obs=len(z_all))

    bank_z = z_all[tr]
    bank_pro = protein_norm[tr]
    bank = FlowMemoryBank(temperature=temperature)
    bank.fit(bank_z, bank_pro)

    n_tr = len(tr)
    ret_tr = bank.retrieve(z_all[tr], k=k, query_bank_indices=np.arange(n_tr))
    ret_va = bank.retrieve(z_all[va], k=k)
    ret_te = bank.retrieve(z_all[te], k=k)

    def gather(ret):
        retrieved_z = bank_z[ret.indices[:, :k_retrieved]]
        sim = ret.mean_similarity if ret.mean_similarity.ndim == 2 else ret.mean_similarity[:, None]
        return ret.protein_prior, ret.reference_feature, retrieved_z, sim

    x0_tr, ref_tr, retz_tr, sim_tr = gather(ret_tr)
    x0_va, ref_va, retz_va, sim_va = gather(ret_va)
    x0_te, ref_te, retz_te, sim_te = gather(ret_te)
    return (x0_tr, ref_tr, retz_tr, sim_tr), (x0_va, ref_va, retz_va, sim_va), (x0_te, ref_te, retz_te, sim_te)


@torch.no_grad()
def predict(model, x0, q, ref, retz, sz, disp, sim, batch_size, device, nfe=8, stochastic=False, cfg_scale=1.0, sigma_clip=1e-3):
    model.eval()
    loader = _to_loader(x0, q, ref, retz, sz, disp, sim, batch_size=batch_size, shuffle=False)
    outs = []
    for x0_b, q_b, ref_b, retz_b, sz_b, disp_b, sim_b in loader:
        pred = integrate_cap_flow(
            model, x0_b.to(device), q_b.to(device), ref_b.to(device), retz_b.to(device),
            sz_b.to(device), disp_b.to(device), sim_b.to(device), nfe=nfe, stochastic=stochastic,
            cfg_scale=cfg_scale, sigma_clip=sigma_clip,
        )
        outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if min(args.flow_epochs, args.eval_every, args.batch_size, args.k, args.k_retrieved) < 1 or any(n < 1 for n in args.nfe):
        raise ValueError("Epochs, intervals, batch sizes, neighbors and NFE must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)

    data = load_cache(cache_dir)
    z_all = data["z_all"]
    neighbor_z = data["neighbor_z"]
    disp = data["displacement"]
    protein_norm = data["protein_norm"]
    protein_names = data["protein_names"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]

    (x0_tr, ref_tr, retz_tr, sim_tr), (x0_va, ref_va, retz_va, sim_va), (x0_te, ref_te, retz_te, sim_te) = \
        retrieve_context(data, args.k, args.k_retrieved, args.retrieval_temperature)

    n_protein = len(protein_names)
    rna_dim = z_all.shape[1]
    n_near = neighbor_z.shape[1] // 2  # L=8, near/far split

    model = CapFlow(
        protein_dim=n_protein, rna_dim=rna_dim,
        hidden_dim=args.flow_hidden_dim, depth=args.flow_depth, n_heads=args.flow_n_heads,
        dropout=args.flow_dropout, n_near=n_near, k_retrieved=args.k_retrieved,
        prior_hidden=args.prior_hidden,
    ).to(device)

    # Fixed source calibration from fitting cells only.
    model.prior_adapter.set_retrieval_mean(x0_tr.mean(axis=0))
    resid = protein_norm[tr] - x0_tr
    resid_var = np.clip(np.var(resid, axis=0), 1e-3, None)
    model.prior_adapter.set_logvar_bias(np.log(resid_var))
    shrink = np.clip(x0_tr.std(axis=0) / (protein_norm[tr].std(axis=0) + 1e-8), 1e-3, None)
    de_shrink = 1.0 + (1.0 / shrink - 1.0)
    model.prior_adapter.set_de_shrink(np.clip(de_shrink, 0.5, 4.0).astype(np.float32))
    print(f"init residual std (per-protein): min={resid_var.min()**0.5:.3f} "
          f"mean={(resid_var.mean())**0.5:.3f} max={resid_var.max()**0.5:.3f}", flush=True)

    loader = _to_loader(
        x0_tr, z_all[tr], ref_tr, retz_tr, neighbor_z[tr], disp[tr], sim_tr, protein_norm[tr],
        batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.flow_lr, weight_decay=args.weight_decay)

    folder = Path(args.cache_dir) / "fold0"
    context = context_for_fold(folder)
    out_dir = (Path(args.output_dir) if args.output_dir else Path(args.cache_dir)) / "fold0" / f"seed{args.seed}"
    out_dir.mkdir(parents=True,exist_ok=True)
    reports = {}
    def report(pred): return score_array(pred, folder, context=context)
    best_val_pcc = -float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, args.flow_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "fm": 0.0, "logvar": 0.0}
        n_batches = 0
        t0 = time.time()
        for x0_b, q_b, ref_b, retz_b, sz_b, disp_b, sim_b, y_b in loader:
            x0_b = x0_b.to(device); q_b = q_b.to(device); ref_b = ref_b.to(device)
            retz_b = retz_b.to(device); sz_b = sz_b.to(device); disp_b = disp_b.to(device)
            sim_b = sim_b.to(device); y_b = y_b.to(device)

            if args.neighbor_mask_prob > 0 and random.random() < args.neighbor_mask_prob:
                m = torch.bernoulli(torch.full((sz_b.shape[0], sz_b.shape[1], 1), 0.5, device=device))
                sz_b = sz_b * m

            loss_out = cap_flow_matching_loss(
                model, x0_b, y_b, q_b, ref_b, retz_b, sz_b, disp_b, sim_b,
                sigma_clip=args.sigma_clip, mask_prob=args.mask_prob,
                cond_drop_prob=args.cond_drop_prob,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss_out.loss.detach())
            totals["fm"] += float(loss_out.fm_loss.detach())
            totals["logvar"] += float(loss_out.mean_logvar.detach())
            n_batches += 1

        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(n_batches, 1),
            "logvar": totals["logvar"] / max(n_batches, 1),
            "seconds": time.time() - t0,
        }
        if epoch % args.eval_every == 0 or epoch <= 2 or epoch == args.flow_epochs:
            val_pred = predict(model, x0_va, z_all[va], ref_va, retz_va, neighbor_z[va], disp[va], sim_va,
                               args.batch_size, device, nfe=8, sigma_clip=args.sigma_clip)
            val_pcc = compute_pcc(val_pred, protein_norm[va], protein_names)[2]
            row["val_pcc_nfe8"] = val_pcc
            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            print(f"epoch={epoch:03d} loss={row['train_loss']:.5f} logvar={row['logvar']:.3f} val_PCC-P@8={val_pcc:.4f}", flush=True)
        else:
            print(f"epoch={epoch:03d} loss={row['train_loss']:.5f} logvar={row['logvar']:.3f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nbest_epoch={best_epoch} best_val_pcc={best_val_pcc:.4f}", flush=True)

    for nfe in sorted(set(args.nfe)):
        for cfg in sorted(set(args.cfg_scale)):
            pred = predict(model, x0_te, z_all[te], ref_te, retz_te, neighbor_z[te], disp[te], sim_te,
                           args.batch_size, device, nfe=nfe, cfg_scale=cfg, sigma_clip=args.sigma_clip)
            m = report(pred)
            tag=f"nfe{nfe}_cfg{cfg:g}"
            reports[tag]=m
            save_predictions(out_dir/f"flow_pred_test_{tag}.npz",pred,folder,metadata={"config":vars(args),"nfe":nfe,"cfg":cfg})
            print(f"NFE={nfe} CFG={cfg} (mean-path) PCC-P={m['pcc_protein_mean']:.4f}", flush=True)

    pred_stoch = predict(model, x0_te, z_all[te], ref_te, retz_te, neighbor_z[te], disp[te], sim_te,
                         args.batch_size, device, nfe=8, stochastic=True, sigma_clip=args.sigma_clip)
    m_stoch = report(pred_stoch)
    print(f"NFE=8 (stochastic) PCC-P={m_stoch['pcc_protein_mean']:.4f}", flush=True)

    # Fixed deployment setting, independent of test-set sweep rankings.
    main_pred=predict(model,x0_te,z_all[te],ref_te,retz_te,neighbor_z[te],disp[te],sim_te,args.batch_size,device,nfe=8,cfg_scale=1.0, sigma_clip=args.sigma_clip)
    mean=model.prior_adapter.retrieval_mean.detach().cpu().numpy()
    scale=model.prior_adapter.de_shrink.detach().cpu().numpy()
    calibrated=mean+(x0_te-mean)*scale
    for kind,values in [('flow',main_pred),('retrieval',x0_te),('calibrated_prior',calibrated)]:
        save_predictions(out_dir/f"{kind}_pred_test.npz",values,folder,metadata={"config":vars(args),"kind":kind,"nfe":8})
        reports[kind]=report(values)
    torch.save(best_state,out_dir/'flow_model.pt')
    write_json(out_dir/'metrics.json',reports)
    write_json(out_dir/'run_config.json',{'arguments':vars(args),'context_fingerprint':context.fingerprint,'prediction_nfe':8})
    ret_te_metrics = report(x0_te)
    print(f"DONE. retrieval PCC-P={ret_te_metrics['pcc_protein_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
