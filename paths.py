"""Central, environment-variable-based path resolution for NicheRFT.

Every machine-specific location used by the helper scripts is read from an
environment variable (documented in README.md) so the code runs on any machine
without editing source files. All variables fall back to a standard layout
rooted at ``NICHERFT_BASE``.

Environment variables
---------------------
NICHERFT_BASE        Project root. Under the original repository layout this
                     directory contains ``methods_model/``, ``outputs_baselines/``
                     and ``nichetrans_spatial_benchmark_bundle/``. Default: the
                     current working directory.
NICHERFT_DATA        Standardized h5ad input directory
                     (default ``$NICHERFT_BASE/methods_model/predata_standardized``).
NICHERFT_BUNDLE      Shared benchmark evaluator directory
                     (default ``$NICHERFT_BASE/nichetrans_spatial_benchmark_bundle``).
NICHERFT_OUTPUTS     Output root ``outputs_baselines``
                     (default ``$NICHERFT_BASE/outputs_baselines``).
NICHERFT_TABLES_DIR  Directory where LaTeX tables are written
                     (default ``$NICHERFT_BASE/methods_model``).
NICHERFT_PYTHON      Python executable used by scripts that launch subprocesses
                     (default: ``sys.executable``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


BASE = _env_path("NICHERFT_BASE", Path.cwd())
DATA = _env_path("NICHERFT_DATA", BASE / "methods_model" / "predata_standardized")
BUNDLE = _env_path("NICHERFT_BUNDLE", BASE / "nichetrans_spatial_benchmark_bundle")
OUTPUTS = _env_path("NICHERFT_OUTPUTS", BASE / "outputs_baselines")
TABLES_DIR = _env_path("NICHERFT_TABLES_DIR", BASE / "methods_model")
PYTHON = os.environ.get("NICHERFT_PYTHON") or sys.executable
