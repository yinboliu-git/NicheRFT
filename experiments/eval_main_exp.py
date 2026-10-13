#!/usr/bin/env python3
"""Evaluation only: retrieval, calibrated mean and flow on the same held-out set.
Never retrains, recalibrates a model, silently changes K, or chooses an NFE using test labels.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_modules.evaluation import context_for_fold,score_artifact
from metrics import METRIC_KEYS,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_dir',required=True);p.add_argument('--run_dir',default=None,help='Run root containing foldN/seedN')
    p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default=None);a=p.parse_args()
    fold=Path(a.cache_dir)/f'fold{a.fold}'
    run=(Path(a.run_dir)/f'fold{a.fold}'/f'seed{a.seed}') if a.run_dir else fold/f'seed{a.seed}'
    ctx=context_for_fold(fold);results={}
    for kind in ('retrieval','calibrated_prior','flow'):
        path=run/f'{kind}_pred_test.npz'
        if not path.exists():raise FileNotFoundError(f'{path}: use finalize_stage3.py or pack_predictions.py for legacy predictions; evaluation does not train models')
        results[kind]=score_artifact(path,fold,context=ctx)
    write_json(a.output or run/'comparison_v2.json',results)
    for key in METRIC_KEYS:print(key,*[results[k][key] for k in results])
if __name__=='__main__':main()
