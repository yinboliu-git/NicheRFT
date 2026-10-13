#!/usr/bin/env python3
"""Package existing predictions with their ACTUAL row, column and training IDs.
Does not infer IDs from a matching shape, refit a model or normalize predictions.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_modules.evaluation import context_for_fold,read_predictions
from metrics import SPACE,ids,matrix

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('fold_dir','predictions','cell_ids','protein_names','fit_cell_ids','validation_cell_ids','space','output'):
        p.add_argument('--'+flag,required=True)
    a=p.parse_args();ctx=context_for_fold(a.fold_dir)
    out=Path(a.output)
    if out.suffix!='.npz':raise ValueError('Output must end in .npz')
    if out.exists():raise FileExistsError(out)
    pred=matrix(np.load(a.predictions))
    fields={key:np.asarray(ids(np.load(getattr(a,key),allow_pickle=True)),dtype=str)
            for key in ('cell_ids','protein_names','fit_cell_ids','validation_cell_ids')}
    out.parent.mkdir(parents=True,exist_ok=True)
    # Validate a temporary artifact before publishing the final one.
    import tempfile,os
    fd,tmp=tempfile.mkstemp(suffix='.npz',dir=out.parent);os.close(fd)
    try:
        np.savez_compressed(tmp,predictions=pred,space=np.asarray(a.space),**fields)
        read_predictions(tmp,ctx)
        os.replace(tmp,out)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
if __name__=='__main__':main()
