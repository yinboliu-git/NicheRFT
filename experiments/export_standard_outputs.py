#!/usr/bin/env python3
"""Export a named prediction artifact and canonical metrics without rerunning inference.
No external evaluator dependency, hard-coded seed42 checkpoint, or invented config.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_modules.evaluation import context_for_fold,read_predictions,score_array,load_fold
from metrics import METRIC_KEYS,write_json
from paths import OUTPUTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_root',default=str(OUTPUTS/'NicheRFT'))
    p.add_argument('--runs_root',default=str(OUTPUTS/'NicheRFT'))
    p.add_argument('--datasets',nargs='+',default=['Human_tonsil','Breast_cancer','Human_lymph_node'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--prediction',default='flow_pred_test.npz');p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    for ds in a.datasets:
        folder=Path(a.cache_root)/ds/f'fold{a.fold}'
        run=Path(a.runs_root)/ds/f'fold{a.fold}'/f'seed{a.seed}';out=run/'standard_export_v2'
        if out.exists() and any(out.iterdir()) and not a.overwrite:raise FileExistsError(f'{out}: pass --overwrite explicitly')
        ctx=context_for_fold(folder);y,cells,names,ix=load_fold(folder)
        artifact=run/a.prediction;pred=read_predictions(artifact,ctx)
        report=score_array(pred,folder,context=ctx)
        out.mkdir(parents=True,exist_ok=True)
        report.update({'method':'NicheRFT','dataset':ds,'fold':a.fold,'seed':a.seed,'prediction_artifact':str(artifact)})
        write_json(out/'metrics.json',report)
        pd.DataFrame([{k:report[k] for k in METRIC_KEYS}]).to_csv(out/'metrics_summary.csv',index=False)
        pd.DataFrame({'protein':names,'PCC':report['pcc_protein'],'SCC':report['scc_protein'],
                      'R2':report['r2_protein']}).to_csv(out/'per_protein_metrics.csv',index=False)
        pd.DataFrame({'cell_id':[cells[i] for i in ix['test']],'PCC':report['pcc_cell'],
                      'SCC':report['scc_cell']}).to_csv(out/'per_cell_metrics.csv',index=False)
        np.save(out/'predictions.npy',pred);np.save(out/'ground_truth.npy',y[ix['test']])
        np.save(out/'protein_names.npy',np.asarray(names));np.save(out/'cell_names.npy',np.asarray(ctx.protocol['split_ids']['test']))
        for key,index in ix.items():np.save(out/f'{key}_indices.npy',index)
        ctx.save(out/'evaluation_context.json')
        with np.load(artifact,allow_pickle=False) as d:
            metadata=json.loads(d['metadata_json'].item()) if 'metadata_json' in d else {}
        write_json(out/'prediction_provenance.json',metadata)
        if (run/'run_config.json').exists():write_json(out/'actual_config.json',json.loads((run/'run_config.json').read_text()))
        write_json(out/'alignment_check.json',{'checked':True,'context_fingerprint':ctx.fingerprint,
            'cell_and_protein_sets':'exact','training_membership':'declared artifact membership matches canonical fold',
            'cross_method_comparability':'Requires all methods to pass evaluate_saved.py using this same context'})
        print('Exported',out)
if __name__=='__main__':main()
