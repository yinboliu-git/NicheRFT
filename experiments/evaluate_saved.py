#!/usr/bin/env python3
"""Score any method's named NPZ artifacts under a single canonical fold.
Example: python experiments/evaluate_saved.py --fold_dir cache/fold0 --predictions A.npz B.npz --output scores.json
External NPZ: predictions, cell_ids, protein_names, fit_cell_ids,
validation_cell_ids, space='log1p_library_10000'. Membership must describe the
actual fitted method. Arrays may be reordered, never silently intersected.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_modules.evaluation import context_for_fold,score_artifact
from metrics import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold_dir',required=True);p.add_argument('--predictions',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['validation','test'],default='test')
    a=p.parse_args();ctx=context_for_fold(a.fold_dir)
    results={str(Path(f).resolve()):score_artifact(f,a.fold_dir,split=a.split,context=ctx) for f in a.predictions}
    write_json(a.output,results)
if __name__=='__main__':main()
