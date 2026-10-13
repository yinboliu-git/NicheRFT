"""NicheRFT canonical metrics, version 2 (common-scale errors, fixed-kernel MMD²).

No prediction-dependent normalization, bandwidth fitting, silent truncation, or
NaN dropping. Full reports require an EvaluationContext fitted on fitting-only
proteins. Cheap PCC is available separately for checkpoint selection.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata

VERSION = 'nicherft-evaluation-v2'
SPACE = 'log1p_library_10000'
EPS = 1e-8
METRIC_KEYS = ('PCC-P','SCC-P','PCC-C','SCC-C','RMSE','MAE','MSE','MMD','CMD-P','CMD-C','R2-P')


def matrix(a, name='array'):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError(f'{name}: require nonempty finite 2-D numeric matrix')
    return a


def pair(pred, true):
    p, y = matrix(pred, 'predictions'), matrix(true, 'targets')
    if p.shape != y.shape:
        raise ValueError(f'Prediction/target shape mismatch: {p.shape} vs {y.shape}')
    return p, y


def ids(values, n=None, name='IDs'):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f'{name}: require 1-D identifiers')
    v = raw.astype(str).tolist()
    if (n is not None and len(v) != n) or len(v) != len(set(v)) or any(not x for x in v):
        raise ValueError(f'{name}: wrong count, empty or duplicate identifiers')
    return v


def digest(a):
    a = np.ascontiguousarray(a, dtype='<f4')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def hash_json(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
        if isinstance(x, dict): return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [convert(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(convert(value),indent=2,allow_nan=False))


@dataclass(frozen=True)
class EvaluationContext:
    """Method-independent protocol. Fingerprint covers labels/IDs/scale/kernel."""
    protocol: dict

    @classmethod
    def fit(cls, fitting_targets, protein_names, fit_cell_ids, *, split_ids, split_targets,
            space=SPACE, seed=42, bandwidth_samples=512, mmd_max_cells=2048):
        y=matrix(fitting_targets,'fitting targets')
        names=ids(protein_names,y.shape[1],'protein names')
        fi=ids(fit_cell_ids,len(y),'fitting cell IDs')
        if len(fi)<2: raise ValueError('At least two fitting cells required for bandwidth')
        if space != SPACE: raise ValueError(f'Expected evaluation space {SPACE}, got {space}')
        if bandwidth_samples<2 or mmd_max_cells<2: raise ValueError('Sample limits must be >=2')
        splits={};hashes={};seen=set(fi)
        if set(split_ids)!=set(split_targets) or not {'validation','test'} <= set(split_ids):
            raise ValueError('Declare validation/test IDs and corresponding targets')
        for key in sorted(split_ids):
            sy=matrix(split_targets[key],key+' targets')
            si=ids(split_ids[key],len(sy),key+' IDs')
            if sy.shape[1]!=len(names) or seen.intersection(si):
                raise ValueError('Overlapping partitions or inconsistent protein panel')
            seen.update(si);splits[key]=si;hashes[key]=digest(sy)
        # Sort by ID before sampling: reproducible under permutations of fitting rows.
        order=np.argsort(np.asarray(fi));rng=np.random.default_rng(seed)
        chosen=order[rng.choice(len(y),min(bandwidth_samples,len(y)),replace=False)]
        distances=pdist(y[chosen],metric='sqeuclidean');positive=distances[distances>0]
        sigma2=float(np.median(positive)) if len(positive) else 1.0
        if not np.isfinite(sigma2) or sigma2<=0: raise ValueError('Invalid fitting bandwidth')
        prot={'version':VERSION,'space':space,'seed':int(seed),'protein_names':names,
              'fit_cell_ids':fi,'fit_target_hash':digest(y),'split_ids':splits,
              'split_target_hashes':hashes,'mmd_sigma2':sigma2,
              'bandwidth_source':'fitting_only','bandwidth_cell_ids':[fi[i] for i in chosen],
              'bandwidth_degenerate':len(positive)==0,'mmd_max_cells':int(mmd_max_cells),
              'mmd_estimator':'biased_empirical_MMD_squared_blockwise',
              'constant_correlation':'zero_if_std_le_1e-8',
              'errors':'common_scale_no_independent_standardization'}
        return cls(prot)

    @property
    def fingerprint(self): return hash_json(self.protocol)

    def save(self,path):
        path=Path(path)
        record={'fingerprint':self.fingerprint,'protocol':self.protocol}
        if path.exists():
            if EvaluationContext.load(path).fingerprint!=self.fingerprint:
                raise ValueError(f'Existing evaluation context differs: {path}; use a new experiment directory')
        else: write_json(path,record)

    @classmethod
    def load(cls,path):
        record=json.loads(Path(path).read_text());obj=cls(record['protocol'])
        if record.get('fingerprint')!=obj.fingerprint or obj.protocol.get('version')!=VERSION:
            raise ValueError('Invalid or obsolete evaluation context')
        return obj

    def check(self,true,protein_names,cell_ids,split,space):
        if space!=self.protocol['space']: raise ValueError('Prediction scale differs from evaluation context')
        if ids(protein_names,true.shape[1],'protein names')!=self.protocol['protein_names']:
            raise ValueError('Protein order/panel mismatch')
        expected=self.protocol['split_ids'].get(split)
        if expected is None or ids(cell_ids,len(true),'cell IDs')!=expected:
            raise ValueError('Cell order/partition mismatch; align by IDs before scoring')
        if digest(true)!=self.protocol['split_target_hashes'][split]:
            raise ValueError('Ground truth differs from canonical context')


def correlations(pred,true,axis=0,rank=False):
    p,y=pair(pred,true)
    if axis not in (0,1): raise ValueError('axis must be 0 or 1')
    if axis==0:p,y=p.T,y.T
    result=np.empty(len(p));valid=np.empty(len(p),bool)
    for start in range(0,len(p),512):
        a=p[start:start+512];b=y[start:start+512]
        # Constant convention is shared by PCC and SCC on original-scale vectors.
        ok=(a.std(axis=1)>EPS)&(b.std(axis=1)>EPS)
        if rank:a=rankdata(a,axis=1,method='average');b=rankdata(b,axis=1,method='average')
        a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        r=np.divide(np.sum(a*b,axis=1),den,out=np.zeros(len(a)),where=ok&(den>0))
        result[start:start+len(a)]=np.clip(r,-1,1);valid[start:start+len(a)]=ok
    return result,valid


def compute_pcc(pred,true,protein_names=None,cell_ids=None):
    p,y=pair(pred,true)
    if protein_names is not None:ids(protein_names,p.shape[1],'protein names')
    if cell_ids is not None:ids(cell_ids,len(p),'cell IDs')
    pp,_=correlations(p,y);pc,_=correlations(p,y,axis=1)
    return pp,pc,float(pp.mean()),float(pc.mean())


def compute_scc(pred,true):
    pp,_=correlations(pred,true,rank=True);pc,_=correlations(pred,true,axis=1,rank=True)
    return pp,pc,float(pp.mean()),float(pc.mean())


def compute_rmse(pred,true):
    p,y=pair(pred,true);return float(np.sqrt(np.mean((p-y)**2)))


def _normalized_rows(x):
    std=x.std(axis=1);x=x-x.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=(std[:,None]>EPS)&(norm>0))


def cmd_dist(A,B):
    A,B=pair(A,B);den=np.linalg.norm(A)*np.linalg.norm(B)
    if den==0:return 0.0 if np.array_equal(A,B) else 1.0
    return float(np.clip(1-np.sum(A*B)/den,0,2))


def _gram_cmd(a,b):
    # Exact identity: <AA^T, BB^T> = ||A^T B||_F². Avoid an N_cells² matrix.
    if a.shape[1]<=a.shape[0]:
        cross=np.sum((a.T@b)**2);aa=np.sum((a.T@a)**2);bb=np.sum((b.T@b)**2)
    else:
        A=a@a.T;B=b@b.T;cross=np.sum(A*B);aa=np.sum(A*A);bb=np.sum(B*B)
    den=np.sqrt(aa*bb)
    if den==0:return 0.0 if aa==bb else 1.0
    return float(np.clip(1-cross/den,0,2))


def compute_cmd(pred,true):
    p,y=pair(pred,true)
    return (_gram_cmd(_normalized_rows(p),_normalized_rows(y)),
            _gram_cmd(_normalized_rows(p.T),_normalized_rows(y.T)))


def _kernel_mean(a,b,sigma2,block_size=256):
    total=0.0
    for i in range(0,len(a),block_size):
        for j in range(0,len(b),block_size):
            total+=np.exp(-cdist(a[i:i+block_size],b[j:j+block_size],'sqeuclidean')/(2*sigma2)).sum()
    return total/(len(a)*len(b))


def compute_mmd(x,y,*,sigma2,indices=None):
    """Biased empirical MMD², fixed RBF k=exp(-||a-b||²/(2*sigma2))."""
    x,y=pair(x,y)
    if not np.isfinite(sigma2) or sigma2<=0:raise ValueError('Positive fixed fitting-set sigma2 required')
    if indices is not None:x,y=x[indices],y[indices]
    val=_kernel_mean(x,x,sigma2)+_kernel_mean(y,y,sigma2)-2*_kernel_mean(x,y,sigma2)
    if val < -1e-10:raise ArithmeticError('Unexpected negative empirical MMD²')
    return float(max(val,0))


def evaluate_predictions(pred,true,protein_names=None,verbose=True,max_eval_cells=None,*,
                         context:EvaluationContext,cell_ids,split='test',space=SPACE):
    if max_eval_cells is not None:raise ValueError('Global metric subsampling is disabled; MMD has a declared separate limit')
    p,y=pair(pred,true);context.check(y,protein_names,cell_ids,split,space)
    pp,pvalid=correlations(p,y);pc,cvalid=correlations(p,y,axis=1)
    sp,_=correlations(p,y,rank=True);sc,_=correlations(p,y,axis=1,rank=True)
    # Same MMD cells for every method/seed in this context. Point metrics use all cells.
    rng=np.random.default_rng(context.protocol['seed'])
    order=np.argsort(np.asarray(cell_ids,dtype=str))
    ix=order[rng.choice(len(p),min(len(p),context.protocol['mmd_max_cells']),replace=False)]
    mmd=compute_mmd(p,y,sigma2=context.protocol['mmd_sigma2'],indices=ix)
    cmd_c,cmd_p=compute_cmd(p,y);error=p-y;mse=float(np.mean(error**2))
    sst=np.sum((y-y.mean(axis=0))**2,axis=0);ok=y.std(axis=0)>EPS
    r2=np.full(y.shape[1],np.nan);r2[ok]=1-np.sum(error[:,ok]**2,axis=0)/sst[ok]
    result={'PCC-P':float(pp.mean()),'SCC-P':float(sp.mean()),'PCC-C':float(pc.mean()),
            'SCC-C':float(sc.mean()),'RMSE':float(np.sqrt(mse)),'MSE':mse,'MAE':float(np.abs(error).mean()),
            'MMD':mmd,'MMD2':mmd,'CMD-P':cmd_p,'CMD-C':cmd_c,
            'R2-P':float(r2[ok].mean()) if ok.any() else None,
            'pcc_protein':pp,'pcc_cell':pc,'scc_protein':sp,'scc_cell':sc,'r2_protein':r2,
            'pcc_protein_median':float(np.median(pp)),'pcc_cell_median':float(np.median(pc)),
            'evaluation':{'version':VERSION,'context_fingerprint':context.fingerprint,'space':space,
                          'split':split,'n_cells':len(p),'n_proteins':p.shape[1],
                          'mmd_sigma2':context.protocol['mmd_sigma2'],'mmd_n_cells':len(ix),
                          'mmd_cell_ids':[cell_ids[i] for i in ix],'mmd_estimator':context.protocol['mmd_estimator'],
                          'undefined_pcc_proteins':int((~pvalid).sum()),'undefined_pcc_cells':int((~cvalid).sum()),
                          'r2_valid_proteins':int(ok.sum())}}
    for alias,key in {'pcc_protein_mean':'PCC-P','pcc_cell_mean':'PCC-C','scc_protein_mean':'SCC-P',
                      'scc_cell_mean':'SCC-C','rmse':'RMSE','mmd':'MMD','cmd_cell':'CMD-C','cmd_protein':'CMD-P'}.items():
        result[alias]=result[key]
    if verbose:print(' '.join(f'{k}={result[k]:.6f}' for k in METRIC_KEYS if result[k] is not None))
    return result
