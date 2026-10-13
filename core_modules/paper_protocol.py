"""Prespecified RNA-to-protein configurations from NicheRFT_submission_final.tex.

These are configurations, not experimental results or evidence of past runs.
"""
from pathlib import Path
import json

PAPER_K = {"tonsil": 32, "lymph": 128, "breast": 2048}
PAPER_SEEDS = (42, 43, 44, 45, 46)
PAPER_NFE = 8
PAPER_SWEEP = (1, 2, 4, 8, 16, 32, 50)


def stage12_command(python, dataset, data_path, cache_root, seed, *, fold=0,
                    split_seed=42, split_manifest=None, flags=()):
    cmd = [str(python), "-u", "experiments/cache_stage12.py", "--dataset", dataset,
           "--data_path", str(data_path), "--output_dir", str(cache_root),
           "--seed", str(seed), "--split_seed", str(split_seed), "--only_fold", str(fold),
           "--k", str(PAPER_K[dataset]), "--retrieval_temperature", "0.35"]
    if split_manifest:
        cmd += ["--split_manifest", str(split_manifest)]
    return cmd + list(flags)


def flow_command(python, dataset, cache_root, run_root, seed, *, fold=0,
                 flags=(), objective="flow", sweep=(8,)):
    return [str(python), "-u", "experiments/finalize_stage3.py", "--cache_dir", str(cache_root),
            "--output_dir", str(run_root), "--seed", str(seed), "--fold", str(fold),
            "--k", str(PAPER_K[dataset]), "--retrieval_temperature", "0.35",
            "--objective", objective, "--prediction_nfe", "8", "--flow_path", "straight",
            "--nfe", *map(str, sweep), *flags]


def check_seed_cache(folder, seed):
    path = Path(folder) / "cache_config.json"
    if not path.exists():
        raise ValueError(f"Missing Stage-I/II provenance: {path}")
    cfg = json.loads(path.read_text())
    if cfg.get("seed") != seed:
        raise ValueError(f"Stage-I/II seed {cfg.get('seed')} differs from run seed {seed}")
    return cfg
