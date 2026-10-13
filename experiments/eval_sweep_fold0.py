"""Select retrieval hyperparameters from validation PCC-P (fold0 only).

Usage:
  python experiments/eval_sweep_fold0.py --sweep_dir outputs/postnatal_sweep
"""
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import compute_pcc
from core_modules.evaluation import context_for_fold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", type=str, required=True)
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results = {}
    context_id = None

    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("bs"):
            continue
        fold0 = d / "fold0"
        x0_file = fold0 / "x0_val.npy"
        pn_file = fold0 / "protein_norm.npy"
        te_file = fold0 / "val_idx.npy"
        names_file = fold0 / "protein_names.npy"
        if not x0_file.exists():
            continue

        ctx=context_for_fold(fold0)
        if context_id is not None and ctx.fingerprint!=context_id:
            raise ValueError('Sweep candidates use different folds/targets/protocols')
        context_id=ctx.fingerprint
        x0 = np.load(x0_file)
        pn = np.load(pn_file)
        va = np.load(te_file)
        names = list(np.load(names_file, allow_pickle=True))

        _, _, pcc_p, _ = compute_pcc(x0, pn[va], names)
        results[d.name] = float(pcc_p)
        print(f"{d.name}: PCC-P={pcc_p:.6f}")

    # Find best
    if not results: raise ValueError("No validation predictions found")
    best = max(results, key=results.get)
    print(f"\nBest: {best} → PCC-P={results[best]:.6f}")

    # Save
    out_file = sweep_dir / "validation_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
