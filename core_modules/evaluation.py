"""Canonical folds, named prediction artifacts, and method-independent evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from metrics import (EvaluationContext, SPACE, VERSION, METRIC_KEYS, ids, digest,
                     pair, evaluate_predictions, write_json)
from core_modules.protocol import validate_three_way_split


def load_fold(folder):
    folder=Path(folder)
    if not (folder/'cell_names.npy').exists():
        raise FileNotFoundError(f'{folder}/cell_names.npy missing. Bind legacy caches explicitly with experiments/bind_legacy_cache.py; do not infer row identity from shape.')
    y=np.load(folder/'protein_norm.npy')
    cells=ids(np.load(folder/'cell_names.npy',allow_pickle=True),len(y),'cell names')
    names=ids(np.load(folder/'protein_names.npy',allow_pickle=True),y.shape[1],'protein names')
    splits={key:np.load(folder/f'{file}_idx.npy') for key,file in [('fit','train'),('validation','val'),('test','test')]}
    validate_three_way_split(splits['fit'],splits['validation'],splits['test'],n_obs=len(y))
    if any(len(v)==0 for v in splits.values()):raise ValueError('All three partitions must be nonempty')
    return y,cells,names,splits


def context_for_fold(folder):
    folder=Path(folder);y,cells,names,ix=load_fold(folder)
    context=EvaluationContext.fit(y[ix['fit']],names,[cells[i] for i in ix['fit']],
        split_ids={k:[cells[i] for i in ix[k]] for k in ('validation','test')},
        split_targets={k:y[ix[k]] for k in ('validation','test')})
    context.save(folder/'evaluation_context.json')
    write_json(folder/'split_manifest.json',{'fit_cell_ids':context.protocol['fit_cell_ids'],
        'validation_cell_ids':context.protocol['split_ids']['validation'],
        'test_cell_ids':context.protocol['split_ids']['test']})
    for key in ('fit','validation','test'):
        np.save(folder/f'{key}_cell_names.npy',np.asarray([cells[i] for i in ix[key]],dtype=str))
    return context


def score_array(pred,folder,*,split='test',context=None):
    y,cells,names,ix=load_fold(folder);context=context or context_for_fold(folder)
    return evaluate_predictions(pred,y[ix[split]],names,verbose=False,context=context,
                                cell_ids=[cells[i] for i in ix[split]],split=split)


def save_predictions(path,pred,folder,*,split='test',metadata=None):
    y,cells,names,ix=load_fold(folder);context=context_for_fold(folder)
    p,_=pair(pred,y[ix[split]])
    # Only call this from a producer that knows its own training/cache provenance.
    np.savez_compressed(path,predictions=p.astype(np.float32),
        cell_ids=np.asarray([cells[i] for i in ix[split]]),protein_names=np.asarray(names),
        fit_cell_ids=np.asarray([cells[i] for i in ix['fit']]),
        validation_cell_ids=np.asarray([cells[i] for i in ix['validation']]),
        space=np.asarray(SPACE),context_fingerprint=np.asarray(context.fingerprint),
        metadata_json=np.asarray(json.dumps(metadata or {},allow_nan=False)))


def read_predictions(path,context,*,split='test'):
    with np.load(path,allow_pickle=False) as d:
        required={'predictions','cell_ids','protein_names','fit_cell_ids','validation_cell_ids','space'}
        if not required<=set(d.files):raise ValueError(f'{path}: missing named artifact fields {required-set(d.files)}')
        p=np.asarray(d['predictions']);ci=ids(d['cell_ids'],len(p),'prediction cells')
        pi=ids(d['protein_names'],p.shape[1],'prediction proteins')
        expected=context.protocol['split_ids'][split];names=context.protocol['protein_names']
        if set(ci)!=set(expected) or set(pi)!=set(names):
            raise ValueError('Prediction cell/protein sets differ; no intersection/truncation is allowed')
        if set(ids(d['fit_cell_ids']))!=set(context.protocol['fit_cell_ids']):
            raise ValueError('Fitting/reference membership differs: scores are not comparable')
        if set(ids(d['validation_cell_ids']))!=set(context.protocol['split_ids']['validation']):
            raise ValueError('Validation membership differs: scores are not comparable')
        if str(d['space'].item())!=context.protocol['space']:raise ValueError('Prediction output scale mismatch')
        if 'context_fingerprint' in d and str(d['context_fingerprint'].item())!=context.fingerprint:
            raise ValueError('Prediction artifact was produced for a different evaluation context')
        row={v:i for i,v in enumerate(ci)};col={v:i for i,v in enumerate(pi)}
        p=p[np.ix_([row[v] for v in expected],[col[v] for v in names])]
        return p


def score_artifact(path,folder,*,split='test',context=None):
    context=context or context_for_fold(folder)
    pred=read_predictions(path,context,split=split)
    return score_array(pred,folder,split=split,context=context)


def summarize_runs(records,seeds):
    if len(records)!=len(seeds) or len(set(seeds))!=len(seeds) or len(seeds)<2:
        raise ValueError('At least two distinct, complete seeds required for sample SD')
    contexts={r['evaluation']['context_fingerprint'] for r in records}
    if len(contexts)!=1:raise ValueError('Cannot aggregate different folds, panels or evaluator protocols')
    result={}
    for key in METRIC_KEYS:
        vals=[r[key] for r in records]
        if any(v is None for v in vals):result[key]={'mean':None,'std':None,'values':vals};continue
        x=np.asarray(vals,dtype=float)
        if not np.isfinite(x).all():raise ValueError(f'Nonfinite seed result: {key}')
        result[key]={'mean':float(x.mean()),'std':float(x.std(ddof=1)),'values':vals}
    return {'version':VERSION,'context_fingerprint':next(iter(contexts)),
            'seeds':list(seeds),'n_seeds':len(seeds),'agg':result}


def declared_split_indices(manifest_path,cell_names):
    """Import a shared protocol by cell ID, before any supervised preprocessing."""
    m=json.loads(Path(manifest_path).read_text());cells=ids(cell_names);lookup={c:i for i,c in enumerate(cells)}
    out=[]
    for key in ('fit_cell_ids','validation_cell_ids','test_cell_ids'):
        group=ids(m[key],name=key)
        if not set(group)<=set(cells):raise ValueError(f'{key}: IDs absent from data')
        out.append(np.asarray([lookup[v] for v in group],dtype=np.int64))
    validate_three_way_split(*out,n_obs=len(cells))
    if any(len(v)==0 for v in out):raise ValueError('Shared partitions must be nonempty')
    if len(set(np.concatenate(out)))!=len(cells):raise ValueError('Shared split must partition all input cells')
    return tuple(out)
