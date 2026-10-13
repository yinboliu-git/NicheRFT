#!/usr/bin/env python3
"""Bind old cache rows to their ORIGINAL standardized input; never manufacture IDs.
Fails if normalized protein values, panel, or dimensions differ. This establishes
cache identity only; it cannot prove how an old model was trained.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import ids,SPACE
from core_modules.evaluation import context_for_fold

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold_dir',required=True)
    p.add_argument('--original_h5ad',required=True);a=p.parse_args()
    import anndata
    from utils import log_normalize
    ad=anndata.read_h5ad(a.original_h5ad);f=Path(a.fold_dir)
    names=ids(np.load(f/'protein_names.npy',allow_pickle=True))
    key='protein_names' if 'protein_names' in ad.uns else 'protein_var_names'
    if names!=ids(ad.uns[key]):raise ValueError('Original protein panel/order differs')
    raw=ad.obsm['protein']
    raw=raw.toarray() if hasattr(raw,'toarray') else np.asarray(raw)
    norm=log_normalize(np.clip(raw,0,None))
    cached=np.load(f/'protein_norm.npy')
    if norm.shape!=cached.shape or not np.allclose(norm,cached,rtol=1e-6,atol=1e-6):
        raise ValueError('Original target rows do not match cache; cannot bind IDs')
    cells=np.asarray(ids(ad.obs_names),dtype=str);dest=f/'cell_names.npy'
    if dest.exists() and not np.array_equal(np.load(dest,allow_pickle=True),cells):
        raise ValueError('Refusing to replace existing cache identity')
    np.save(dest,cells);context_for_fold(f)
if __name__=='__main__':main()
