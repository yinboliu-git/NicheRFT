# Quick-start example

A self-contained, end-to-end run of NicheRFT on a small **synthetic** dataset.
No real tissue data and no identifying information are needed: the example data
is generated from a fixed seed by `make_example_data.py`.

The synthetic data mimics a two-slice cross-slice task. Both slices (RNA counts,
a protein panel, spatial coordinates) are drawn from a shared smooth latent
program, so a model that recovers the program can transfer the retrieval prior
from the training slice to the held-out slice.

## Requirements

```bash
pip install -r requirements.txt
```

CPU is sufficient; the script falls back to CPU automatically when no GPU is
visible.

## Run

```bash
bash examples/run_example.sh
# or pick an interpreter
PYTHON=python3 bash examples/run_example.sh
```

The script performs three steps:

1. **Generate** the synthetic dataset into `examples/data/example.h5ad` if it is
   not already present.
2. **Stage I/II** — `experiments/cache_stage12.py`: learn the niche-aware RNA
   representation and cache split-aware retrieval inputs. The retrieval bank
   holds fitting cells only; validation/test cells are queries.
3. **Stage III** — `experiments/finalize_stage3.py`: fit the DiT conditional
   flow-matching vector field (8-step ODE) and evaluate the validation-selected
   checkpoint.

## Output

Everything lands under `outputs/example_flow/fold0/seed42/`:

- `flow_pred_test.npy`, `retrieval_pred_test.npy`, `calibrated_prior_pred_test.npy`
- `flow_pred_test.npz`, ... (named evaluation artifacts with cell IDs and context)
- `flow_model.pt`, `run_config.json`, `metrics.json`

`metrics.json` reports the retrieval prior and the flow against the canonical
evaluator. On the synthetic example the 8-step flow (`NFE=8`, the default
inference) improves on the retrieval prior on every reported axis:

| method | PCC-P | MMD | CMD-P | RMSE |
|---|---|---|---|---|
| retrieval prior | 0.7107 | 0.0447 | 0.0979 | 1.9008 |
| flow, NFE=8 (default) | 0.7241 | 0.0087 | 0.0566 | 1.6599 |

Representative single run (seed 42); Stage-I/III training on this small
synthetic set shows only third-decimal run-to-run variation.

## Notes

- **Small inputs need a small batch.** The example passes `--batch_size 32`;
  the default `1024` collapses to a single optimization step per epoch on a
  dataset this small and under-trains the representation.
- **The default run is fully inductive** (offline): no statistics from the
  query slice are used at inference. An optional flag `--slice_relative_center`
  (Stage III) instead calibrates the de-shrink source around each slice's own
  retrieval-prior mean — still label-free, but transductive (it couples the
  query-slice cells). Add it to `finalize_stage3.py` to enable that variant.
- **Re-running** is safe: the script clears `outputs/example_cache` and
  `outputs/example_flow` at the start (both stages otherwise refuse to
  overwrite an existing cache).
- **Your own data.** Provide an `.h5ad` with the schema written by
  `make_example_data.py` (see its docstring) and point `--data_path` at it,
  choosing the closest `--dataset` preset (or add your own in
  `pipeline/config.py`).
