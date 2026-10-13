#!/usr/bin/env python3
"""Recompute complete seed results using one fixed evaluation context.
Missing seeds fail explicitly; SD is calculated independently for each metric.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_modules.evaluation import context_for_fold,score_artifact,summarize_runs
from metrics import write_json
from paths import OUTPUTS
from core_modules.paper_protocol import check_seed_cache

DATASETS=('Human_tonsil','Breast_cancer','Human_lymph_node')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_root',default=str(OUTPUTS/'NicheRFT'))
    p.add_argument('--runs_root',default=str(OUTPUTS/'NicheRFT_multiseed_v2'))
    p.add_argument('--datasets',nargs='+',default=list(DATASETS));p.add_argument('--fold',type=int,default=0)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--prediction',default='flow_pred_test.npz');p.add_argument('--output',default=None)
    p.add_argument('--cache_layout',choices=['shared','per_seed'],default='shared',
                   help='per_seed is required for full Stage-I--III variability; shared is Stage-III-only')
    a=p.parse_args();results={}
    for ds in a.datasets:
        records=[]
        for seed in a.seeds:
            folder=Path(a.cache_root)/ds
            if a.cache_layout=='per_seed':folder=folder/f'seed{seed}'
            folder=folder/f'fold{a.fold}'
            if a.cache_layout=='per_seed':check_seed_cache(folder,seed)
            ctx=context_for_fold(folder)
            path=Path(a.runs_root)/ds/f'fold{a.fold}'/f'seed{seed}'/a.prediction
            if not path.exists():raise FileNotFoundError(f'Missing seed {seed}: {path}; no partial five-seed summary will be written')
            if a.cache_layout=='per_seed':
                rc=json.loads((path.parent/'run_config.json').read_text())
                expected=hashlib.sha256((folder/'stage1_state.pt').read_bytes()).hexdigest()
                if rc.get('stage1_checkpoint_hash')!=expected or rc.get('stage12_config',{}).get('seed')!=seed:
                    raise ValueError('Run/cache Stage-I provenance differs; cannot claim full-pipeline repetition')
            record=score_artifact(path,folder,context=ctx);records.append(record)
        results[ds]=summarize_runs(records,a.seeds)
        results[ds]['replication_scope']='Stage-I--III' if a.cache_layout=='per_seed' else 'Stage-III-only, fixed Stage-I/II cache'
        for key,v in results[ds]['agg'].items():print(ds,key,v['mean'],v['std'])
    write_json(a.output or Path(a.runs_root)/'multiseed_aggregate_v2.json',results)
if __name__=='__main__':main()
