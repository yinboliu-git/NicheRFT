"""Numerical and protocol checks requiring only NumPy/SciPy (no GPU/torch).

Run: python3 -m unittest discover -s tests -p '*unittest.py' -v
"""
import json
import tempfile
import unittest
import subprocess
import sys
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from metrics import EvaluationContext, compute_mmd, compute_rmse, compute_cmd, correlations
from core_modules.evaluation import context_for_fold, save_predictions, score_artifact, summarize_runs
from core_modules.protocol import split_reference
from core_modules.paper_protocol import stage12_command, flow_command, check_seed_cache


class PaperProtocolTests(unittest.TestCase):
    def test_split_is_separate_from_training_seed(self):
        for training_seed in range(42, 47):
            tr, va = split_reference(np.arange(80), split_seed=42)
            np.testing.assert_array_equal(va, split_reference(np.arange(80), split_seed=42)[1])
            self.assertEqual(len(va), 10)
            cmd = stage12_command("python", "lymph", "data", "cache", training_seed)
            self.assertEqual(cmd[cmd.index("--split_seed") + 1], "42")
            self.assertEqual(cmd[cmd.index("--seed") + 1], str(training_seed))

    def test_paper_presets(self):
        for ds, k in [("tonsil",32),("lymph",128),("breast",2048)]:
            cmd = flow_command("python", ds, "cache", "runs", 42)
            self.assertEqual(cmd[cmd.index("--k") + 1], str(k))
            self.assertEqual(cmd[cmd.index("--prediction_nfe") + 1], "8")
            self.assertEqual(cmd[cmd.index("--flow_path") + 1], "straight")

    def test_biased_mmd_matches_dense_formula(self):
        x=np.array([[0.,1.],[1.,3.],[4.,2.]])
        y=np.array([[2.,0.],[1.,2.],[3.,1.]])
        kernel=lambda a,b: np.exp(-cdist(a,b,'sqeuclidean')/5.)
        expected=kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()
        self.assertAlmostEqual(compute_mmd(x,y,sigma2=2.5), expected)
        self.assertAlmostEqual(compute_mmd(x,x,sigma2=2.5), 0.)

    def test_mmd_requires_fixed_bandwidth(self):
        with self.assertRaises(ValueError):compute_mmd(np.ones((3,2)),np.ones((3,2)),sigma2=0)

    def test_rmse_does_not_independently_standardize(self):
        y=np.arange(8).reshape(4,2)
        self.assertAlmostEqual(compute_rmse(y+2,y),2.)

    def test_correlations_match_scipy_and_constant_convention(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(12,4));y=rng.normal(size=(12,4))
        p,_=correlations(x,y);s,_=correlations(x,y,rank=True)
        np.testing.assert_allclose(p,[pearsonr(x[:,j],y[:,j]).statistic for j in range(4)])
        np.testing.assert_allclose(s,[spearmanr(x[:,j],y[:,j]).statistic for j in range(4)])
        self.assertTrue(np.all(correlations(np.ones_like(x),y)[0]==0))

    def test_positive_affine_preserves_protein_correlations(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(30,4));y=rng.normal(size=(30,4))
        np.testing.assert_allclose(correlations(x,y)[0],correlations(x*np.array([.5,1,2,4])+3,y)[0])

    def test_cmd_matches_explicit_matrices(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(12,4));y=rng.normal(size=(12,4))
        values=compute_cmd(x,y)
        for axis,val in zip([True,False],values):
            A=np.corrcoef(x,rowvar=axis);B=np.corrcoef(y,rowvar=axis)
            expected=1-np.sum(A*B)/(np.linalg.norm(A)*np.linalg.norm(B))
            self.assertAlmostEqual(val,expected)

    def test_bandwidth_never_uses_test_targets(self):
        fit=np.arange(16).reshape(8,2);val=np.array([[1.,4.],[3.,6.]]);test=np.array([[2.,5.],[4.,7.]])
        def ctx(test):return EvaluationContext.fit(fit,['a','b'],[f'f{i}' for i in range(8)],
            split_ids={'validation':['v1','v2'],'test':['t1','t2']},split_targets={'validation':val,'test':test})
        self.assertEqual(ctx(test).protocol['mmd_sigma2'],ctx(test*100).protocol['mmd_sigma2'])

    def test_seed_summary_uses_sample_sd_and_rejects_mixed_contexts(self):
        from metrics import METRIC_KEYS
        records=[{**{k:v for k in METRIC_KEYS},'evaluation':{'context_fingerprint':'same'}} for v in [.1,.3,.5,.7,.9]]
        result=summarize_runs(records,list(range(42,47)))
        self.assertAlmostEqual(result['agg']['PCC-P']['std'],np.std([.1,.3,.5,.7,.9],ddof=1))
        records[-1]['evaluation']['context_fingerprint']='different'
        with self.assertRaises(ValueError):summarize_runs(records,list(range(42,47)))

    def test_named_prediction_roundtrip_and_seed_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);rng=np.random.default_rng(0);y=rng.normal(size=(16,3)).astype('float32')
            for name,a in [('protein_norm',y),('protein_names',np.array(['a','b','c'])),
                ('cell_names',np.array([f'c{i}' for i in range(16)])),('train_idx',np.arange(10)),
                ('val_idx',np.arange(10,12)),('test_idx',np.arange(12,16))]:np.save(f/(name+'.npy'),a)
            context_for_fold(f);save_predictions(f/'prediction.npz',y[12:],f)
            r=score_artifact(f/'prediction.npz',f)
            self.assertAlmostEqual(r['MMD'],0.);self.assertAlmostEqual(r['PCC-P'],1.)
            (f/'cache_config.json').write_text(json.dumps({'seed':43}))
            check_seed_cache(f,43)
            with self.assertRaises(ValueError):check_seed_cache(f,42)

    def test_per_seed_cli_aggregation_and_missing_seed_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.default_rng(20);y=rng.normal(size=(16,3)).astype('float32')
            for seed in [42,43]:
                folder=root/'caches'/'lymph'/f'seed{seed}'/'fold0';folder.mkdir(parents=True)
                run=root/'runs'/'lymph'/'fold0'/f'seed{seed}';run.mkdir(parents=True)
                for name,a in [('protein_norm',y),('protein_names',np.array(['a','b','c'])),
                    ('cell_names',np.array([f'c{i}' for i in range(16)])),('train_idx',np.arange(10)),
                    ('val_idx',np.arange(10,12)),('test_idx',np.arange(12,16))]:np.save(folder/(name+'.npy'),a)
                (folder/'cache_config.json').write_text(json.dumps({'seed':seed,'split_seed':42}))
                checkpoint=f'synthetic fixture checkpoint for {seed}'.encode()
                (folder/'stage1_state.pt').write_bytes(checkpoint)
                (run/'run_config.json').write_text(json.dumps({'stage12_config':{'seed':seed},
                    'stage1_checkpoint_hash':hashlib.sha256(checkpoint).hexdigest()}))
                context_for_fold(folder)
                save_predictions(run/'flow_pred_test.npz',y[12:]+(seed-42)*.1,folder)
            repo=Path(__file__).resolve().parents[1]
            cmd=[sys.executable,str(repo/'experiments/eval_multiseed.py'),'--cache_root',str(root/'caches'),
                 '--runs_root',str(root/'runs'),'--cache_layout','per_seed','--datasets','lymph','--seeds','42','43']
            subprocess.run(cmd,cwd=repo,capture_output=True,text=True,check=True)
            result=json.loads((root/'runs'/'multiseed_aggregate_v2.json').read_text())
            self.assertEqual(result['lymph']['replication_scope'],'Stage-I--III')
            self.assertEqual(result['lymph']['n_seeds'],2)
            bad=subprocess.run(cmd+['44'],cwd=repo,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)

    def test_training_self_exclusion_and_query_reference_limits(self):
        from core_modules.flow_retrieval import FlowMemoryBank
        z=np.eye(6,dtype='float32');p=np.arange(12,dtype='float32').reshape(6,2)
        bank=FlowMemoryBank(use_faiss=False);bank.fit(z,p)
        retrieval=bank.retrieve(z,k=2,query_bank_indices=np.arange(6))
        for i in range(6):self.assertNotIn(i,retrieval.indices[i])
        query=bank.retrieve(z[:2]+.1,k=2)
        self.assertLess(query.indices.max(),6)


if __name__ == '__main__':
    unittest.main()
