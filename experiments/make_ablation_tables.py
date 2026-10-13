#!/usr/bin/env python3
"""Render NicheRFT ablation results.json -> TableA1.md (appendix) + TableA2.md (main).

Reads outputs_baselines/NicheRFT_ablation/{tonsil,lymph,breast}/results.json and emits
two LaTeX tables plus a printed niche-contribution summary.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import OUTPUTS, TABLES_DIR

OUT = OUTPUTS / "NicheRFT_ablation_v2"
DST = TABLES_DIR

DATASETS = ["tonsil", "lymph", "breast"]
DISPLAY = {"tonsil": "Tonsil", "lymph": "Lymph", "breast": "Breast"}

# TableA2 (main-text niche decomposition): variant -> label
A2_ROWS = [
    ("full", "Full NicheRFT (control)"),
    ("nonich_enc", "w/o niche branch (encoder)"),
    ("nospatial", "w/o spatial tokens (flow)"),
    ("nonich_full", "w/o niche (fully non-spatial)"),
    ("retrieval_only", "Retrieval prior only"),
]

# TableA1 (appendix, grouped): (stage, variant, label)
A1_ROWS = [
    ("Stage 1", "nonich_enc", "w/o niche branch"),
    ("Stage 1", "nocontrastive", "w/o contrastive ($\\alpha=0$)"),
    ("Stage 1", "nozinb", "w/o ZINB reconstruction ($\\beta=\\gamma=0$)"),
    ("Stage 2", "k1", "Single-neighbor retrieval ($K=1$)"),
    ("Stage 2", "uniform", "Uniform neighbor aggregation"),
    ("Stage 3", "retrieval_only", "Retrieval prior only (w/o Stage-3)"),
    ("Stage 3", "nodedeshrink", "w/o de-shrink"),
    ("Stage 3", "calibrated_prior", "Calibrated prior only"),
    ("Stage 3", "nospatial", "w/o spatial tokens"),
    ("Stage 3", "gaussian", "w/o retrieval (Gaussian source)"),
    ("Control", "full", "Full NicheRFT (control)"),
]


def load() -> dict[str, dict[str, dict[str, float]]]:
    res = {}
    for ds in DATASETS:
        p = OUT / ds / "results.json"
        res[ds] = json.loads(p.read_text())
    return res


def fmt(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


def main() -> None:
    res = load()
    DST.mkdir(parents=True,exist_ok=True)
    from metrics import VERSION
    for ds, rows in res.items():
        protocols={v.get('evaluation',{}).get('context_fingerprint') for v in rows.values()}
        if len(protocols)!=1 or None in protocols or any(v.get('evaluation',{}).get('version')!=VERSION for v in rows.values()):
            raise ValueError(f'{ds}: mixed or legacy evaluation results; recompute before rendering tables')

    contrib = {}
    for ds in DATASETS:
        full = res[ds]["full"]
        nn = res[ds]["nonich_full"]
        contrib[ds] = {
            "dPCCP": full["PCC-P"] - nn["PCC-P"],
            "dMMD": nn["MMD"] - full["MMD"],  # positive = niche lowers MMD
            "dPCCC": full["PCC-C"] - nn["PCC-C"],
        }
    print("Niche contribution (full - fully-non-spatial):")
    for ds in DATASETS:
        c = contrib[ds]
        print(f"  {DISPLAY[ds]:8s} dPCC-P={c['dPCCP']:+.4f}  dMMD={c['dMMD']:+.4f}  dPCC-C={c['dPCCC']:+.4f}")
    best = max(DATASETS, key=lambda d: contrib[d]["dPCCP"])
    print(f"  -> largest PCC-P niche gain: {DISPLAY[best]}\n")

    flow_delta = {}
    for ds in DATASETS:
        f = res[ds]["full"]; r = res[ds]["retrieval_only"]
        flow_delta[ds] = {
            "dPCCP": f["PCC-P"] - r["PCC-P"],   # positive = flow improves point prediction
            "dMMD": f["MMD"] - r["MMD"],        # negative = flow lowers MMD (distribution recovery)
        }
    print("Combined calibration and flow change (full - retrieval prior):")
    for ds in DATASETS:
        c = flow_delta[ds]
        print(f"  {DISPLAY[ds]:8s} dPCC-P={c['dPCCP']:+.4f}  dMMD={c['dMMD']:+.4f}")
    print()

    def cells(variant: str) -> str:
        out = []
        for ds in DATASETS:
            r = res[ds][variant]
            out.append(f"{fmt(r['PCC-P'])} & {fmt(r['MMD'])}")
        return " & ".join(out)

    def flow_delta_cells() -> str:
        out = []
        for ds in DATASETS:
            c = flow_delta[ds]
            out.append(f"{c['dPCCP']:+.4f} & {c['dMMD']:+.4f}")
        return " & ".join(out)

    # ---------------- Table A2 ----------------
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l cc|cc|cc @{}}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{2}{c}{Tonsil} & \\multicolumn{2}{c}{Lymph} & \\multicolumn{2}{c}{Breast} \\\\")
    lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
    lines.append("Ablation variant & PCC-P$\\uparrow$ & MMD$\\downarrow$ & PCC-P$\\uparrow$ & MMD$\\downarrow$ & PCC-P$\\uparrow$ & MMD$\\downarrow$ \\\\")
    lines.append("\\midrule")
    for variant, label in A2_ROWS:
        bold = "\\textbf{" if variant == "full" else ""
        bold_e = "}" if variant == "full" else ""
        lines.append(f"{bold}{label}{bold_e} & {cells(variant)} \\\\")
    lines.append("\\midrule")
    d = []
    for ds in DATASETS:
        c = contrib[ds]
        d.append(f"{c['dPCCP']:+.4f} & {c['dMMD']:+.4f}")
    lines.append("$\\Delta$ niche (full $-$ non-spatial) & " + " & ".join(d) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\caption{")
    lines.append("Niche-contribution decomposition. Removing the niche branch (encoder) and/or the")
    lines.append("spatial tokens (flow) isolates the microenvironment's contribution to protein")
    lines.append("prediction. $\\Delta$ niche is the gain of the full model over the fully")
    lines.append(f"non-spatial variant; the largest PCC-P gain is on \\textbf{{{DISPLAY[best]}}}.")
    lines.append("Single-seed (42) results in the $\\log1p$-library benchmark space.")
    lines.append("}")
    lines.append("\\label{tab:niche_decomposition}")
    lines.append("\\end{table*}")
    (DST / "TableA2.md").write_text("\n".join(lines) + "\n")

    # ---------------- Table A1 ----------------
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} ll cc|cc|cc @{}}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{Component} & \\multirow{2}{*}{Ablation variant} & "
                 "\\multicolumn{2}{c}{Tonsil} & \\multicolumn{2}{c}{Lymph} & \\multicolumn{2}{c}{Breast} \\\\")
    lines.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    lines.append(" & & PCC-P$\\uparrow$ & MMD$\\downarrow$ & PCC-P$\\uparrow$ & MMD$\\downarrow$ & PCC-P$\\uparrow$ & MMD$\\downarrow$ \\\\")
    lines.append("\\midrule")
    cur_stage = None
    for i, (stage, variant, label) in enumerate(A1_ROWS):
        if stage != cur_stage:
            if cur_stage is not None:
                lines.append("\\midrule")
            if stage == "Control":
                lines.append("  &  $\\Delta$ refinement (full $-$ retrieval prior) & "
                             + flow_delta_cells() + " \\\\")
                lines.append("\\midrule")
            cur_stage = stage
            n = sum(1 for s, _, _ in A1_ROWS if s == stage)
            lines.append(f"\\multirow{{{n}}}{{*}}{{{stage}}}")
        bold = "\\textbf{" if variant == "full" else ""
        bold_e = "}" if variant == "full" else ""
        lines.append(f"  & {bold}{label}{bold_e} & {cells(variant)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\caption{")
    lines.append("Comprehensive component ablation of NicheRFT across three spatial datasets,")
    lines.append("with the full model serving as the control. Single-seed (42) results in the")
    lines.append("$\\log1p$-library benchmark space; PCC-P higher is better, MMD lower is better.")
    lines.append("The  $\\Delta$ refinement row (full minus retrieval prior) includes calibration and flow refinement.")
    lines.append("Interpret the measured signs of the differences; no improvement is assumed in advance.")
    lines.append("}")
    lines.append("\\label{tab:niche_comprehensive_ablation}")
    lines.append("\\end{table*}")
    (DST / "TableA1.md").write_text("\n".join(lines) + "\n")

    print("wrote TableA1.md and TableA2.md")


if __name__ == "__main__":
    main()
