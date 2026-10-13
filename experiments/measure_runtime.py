#!/usr/bin/env python3
"""Measure wall-clock + resource usage of the NicheRFT full pipeline.

Runs Stage1+2 (cache_stage12.py) then Stage3 (finalize_stage3.py) as subprocesses
and samples GPU device/process memory + process RSS, writing a
runtime_resources.json with the same field names as the baseline bundle's
resource_monitor.finalize_resource_report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paths import PYTHON
import numpy as np


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def _nvidia_smi(args):
    try:
        cp = subprocess.run(["nvidia-smi", *args], text=True, capture_output=True, timeout=2, check=False)
        return cp.stdout.strip() if cp.returncode == 0 else ""
    except Exception:
        return ""


def _gpu_device_mem(gpu_id: int) -> float:
    txt = _nvidia_smi(["-i", str(gpu_id), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(txt.splitlines()[0].strip())
    except Exception:
        return float("nan")


def _gpu_process_mem(pid: int) -> float:
    txt = _nvidia_smi(["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    vals = []
    for line in txt.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            try:
                if int(parts[0]) == pid:
                    vals.append(float(parts[1]))
            except Exception:
                pass
    return float(sum(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["breast", "lymph", "tonsil"])
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True, help="scratch cache dir for the fresh run")
    p.add_argument("--report-json", required=True, help="path to write runtime_resources.json")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--n-test", type=int, default=None, help="Optional expected test count; checked against the cache")
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--retrieval-temperature", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--python", default=None, help="python executable (default: nicheflow_env python)")
    args = p.parse_args()

    python = args.python or PYTHON
    exp = REPO / "experiments"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    stages = [
        ("stage12", [python, "-u", str(exp / "cache_stage12.py"),
                     "--dataset", args.dataset, "--data_path", args.data_path,
                     "--output_dir", str(out), "--device", "cuda",
                     "--seed", str(args.seed), "--n_top_genes", "3000",
                     "--n_per_protein", "100", "--alpha", "20.0",
                     "--spatial_temperature", "2.0", "--batch_size", str(args.batch_size),
                     "--k", str(args.k), "--retrieval_temperature", str(args.retrieval_temperature),
                     "--stage1_epochs", str(args.stage1_epochs)]),
        ("stage3", [python, "-u", str(exp / "finalize_stage3.py"),
                    "--cache_dir", str(out), "--output_dir", str(out), "--device", "cuda",
                    "--k", str(args.k), "--retrieval_temperature", str(args.retrieval_temperature),
                    "--seed", str(args.seed), "--fold", "0"]),
    ]

    if args.split_manifest:
        stages[0][1].extend(["--split_manifest", args.split_manifest])

    peak_rss = float("nan")
    peak_gpu_dev = float("nan")
    peak_gpu_proc = float("nan")
    stop = threading.Event()
    samples_lock = threading.Lock()

    def sampler(pid_holder):
        nonlocal peak_rss, peak_gpu_dev, peak_gpu_proc
        while not stop.is_set():
            pid = pid_holder[0]
            rss = _rss_mb(pid) if pid else float("nan")
            dev = _gpu_device_mem(args.gpu)
            proc = _gpu_process_mem(pid) if pid else float("nan")
            with samples_lock:
                if rss == rss and (peak_rss != peak_rss or rss > peak_rss):
                    peak_rss = rss
                if dev == dev and (peak_gpu_dev != peak_gpu_dev or dev > peak_gpu_dev):
                    peak_gpu_dev = dev
                if proc == proc and (peak_gpu_proc != peak_gpu_proc or proc > peak_gpu_proc):
                    peak_gpu_proc = proc
            stop.wait(0.5)

    pid_holder = [None]
    t = threading.Thread(target=sampler, args=(pid_holder,), daemon=True)
    start = time.perf_counter()
    t.start()

    for name, cmd in stages:
        print(f"[runtime] === {name} === {cmd[0]} {' '.join(cmd[2:])}", flush=True)
        proc = subprocess.Popen(cmd, env=env, cwd=str(REPO))
        pid_holder[0] = proc.pid
        rc = proc.wait()
        if rc != 0:
            print(f"[runtime] {name} FAILED with exit code {rc}", flush=True)
            stop.set(); t.join(timeout=2)
            raise SystemExit(rc)

    stop.set(); t.join(timeout=2)
    total = time.perf_counter() - start

    n_test = len(np.load(out / "fold0" / "test_idx.npy"))
    n_train = len(np.load(out / "fold0" / "train_idx.npy"))
    if args.n_test is not None and args.n_test != n_test:
        raise ValueError("--n-test does not match the actual cached test partition")
    report = {
        "Method": "NicheRFT",
        "Dataset": args.dataset,
        "Completion": "COMPLETED",
        "N train": n_train,
        "N test": n_test,
        "Total time (s)": round(total, 4),
        "Total time (min)": round(total / 60.0, 6),
        "Pipeline amortized time (s/test cell; includes training)": round(total / max(1, n_test), 6),
        "Pipeline amortized cells/s (includes training)": round(n_test / total, 4) if total > 0 else None,
        "GPU-hours": round(total / 3600.0, 6),
        "Peak process RSS (MB)": round(peak_rss, 1) if peak_rss == peak_rss else None,
        "Peak GPU process memory (MB)": round(peak_gpu_proc, 1) if peak_gpu_proc == peak_gpu_proc else None,
        "Peak GPU device memory (MB)": round(peak_gpu_dev, 1) if peak_gpu_dev == peak_gpu_dev else None,
        "Confidence": "high" if peak_gpu_proc == peak_gpu_proc else "medium",
        "Spatial input": "official niche neighbors",
        "Note": "Full pipeline = cache_stage12.py + finalize_stage3.py, including training, validation, evaluation and artifact writes; not inference-only throughput",
    }
    rp = Path(args.report_json)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[runtime] DONE total={total:.1f}s peak_rss={report['Peak process RSS (MB)']} peak_gpu_dev={report['Peak GPU device memory (MB)']} peak_gpu_proc={report['Peak GPU process memory (MB)']}", flush=True)
    print(f"[runtime] wrote {rp}", flush=True)


if __name__ == "__main__":
    main()
