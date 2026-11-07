"""Gate inputs, nested source isolation, and fusion bounds."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
from scale_selector import FEATURES,METADATA_FEATURES,feature_map,fuse
import validate_scale_selector as runner


class SelectorTests(unittest.TestCase):
    def setup_arrays(self):
        x=np.random.default_rng(2).random((4,32,32),dtype=np.float32)
        p=np.full((32,32),.2,np.float32);q=np.full((32,32),.8,np.float32)
        return x,p,q

    def coefficients(self):
        return {'features':list(FEATURES),'mean':[0.]*18,'std':[1.]*18,'weights':[0.]*18,'bias':0.}

    def test_meta_fit_and_calibration_disjoint_complete(self):
        for g,n in enumerate([256,794,256,256,256,256]):
            fit,cal=runner.meta_split(n,g)
            self.assertFalse(set(fit)&set(cal))
            self.assertEqual(set(fit)|set(cal),set(range(n)))
            self.assertTrue(np.array_equal(runner.meta_split(n,g)[0],fit))

    def test_target_rejected_for_head_learning(self):
        with self.assertRaises(ValueError):runner.source_keys(range(6),2)
        runner.source_keys([0,1,3,4,5],2)

    def test_zero_gate_is_exact_mean(self):
        x,p,q=self.setup_arrays();f=feature_map(x,p,q,3)
        pred,a=fuse(f,p,q,self.coefficients())
        np.testing.assert_array_equal(a,.5*np.ones_like(p))
        np.testing.assert_array_equal(pred,.5*(p+q))

    def test_gate_cannot_invent_probability_outside_branch_interval(self):
        x,p,q=self.setup_arrays();f=feature_map(x,p,q,3);c=self.coefficients()
        c['weights']=np.linspace(-5,5,18).tolist()
        pred,a=fuse(f,p,q,c)
        self.assertTrue(np.all(pred>=p-1e-7) and np.all(pred<=q+1e-7))

    def test_constant_gate_ignores_all_input_features(self):
        x,p,q=self.setup_arrays();f=feature_map(x,p,q,3);c=self.coefficients();c['bias']=1.4
        a=fuse(f,p,q,c)[0];b=fuse(f*8,p,q,c)[0]
        np.testing.assert_array_equal(a,b)

    def test_no_gsd_ablation_ignores_metadata(self):
        x,p,q=self.setup_arrays();f=feature_map(x,p,q,3);c=self.coefficients()
        c['weights']=[.1]*16+[0.,0.]
        changed=f.copy();changed[...,list(METADATA_FEATURES)]=9
        np.testing.assert_array_equal(fuse(f,p,q,c)[0],fuse(changed,p,q,c)[0])

    def test_ground_truth_is_not_an_inference_feature_argument(self):
        import inspect
        names=list(inspect.signature(feature_map).parameters)
        self.assertEqual(names,['image_chw','global_probability','local_probability','nominal_gsd'])

    def test_invalid_gsd_and_nonfinite_probabilities_rejected(self):
        x,p,q=self.setup_arrays()
        with self.assertRaises(ValueError):feature_map(x,p,q,0)
        p[0,0]=np.nan
        with self.assertRaises(ValueError):feature_map(x,p,q,3)

    def test_nonfinite_fitted_parameters_rejected(self):
        x,p,q=self.setup_arrays();f=feature_map(x,p,q,3);c=self.coefficients();c['bias']=float('nan')
        with self.assertRaises(ValueError):fuse(f,p,q,c)

    def test_positive_sampling_weights_conserve_parent_mass(self):
        import cross_region_scale as base
        x,p,q=self.setup_arrays();gt=np.zeros((32,32),np.uint8);gt[10:12,10:12]=1
        c=base.gt_components(gt,3)
        pack=runner.sampled_parent(feature_map(x,p,q,3),p,q,gt,c,np.random.default_rng(0))
        self.assertAlmostEqual(float(pack[4].sum()),1+3*4/1024,places=6)

    def test_optimizer_learns_conditional_toy_that_constant_cannot_solve(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA needed by actual head fitter')
        pools={}
        for g in range(1,6):
            truth=np.tile(np.array([0.,1.],np.float32),128)
            features=np.zeros((256,18),np.float32);features[:,13]=truth
            pools[g]=[features,np.full(256,.9,np.float32),np.full(256,.1,np.float32),truth,np.ones(256,np.float32)]
        config={'head_steps':120,'head_per_region_batch':64,'head_optimizer':{'lr':.05},'head_regularization':.001}
        c,ct=runner.fit_gate(pools,0,42,'constant_gate',config)
        f,ft=runner.fit_gate(pools,0,42,'feature_gate',config)
        self.assertEqual(ct['sample_schedule_sha256'],ft['sample_schedule_sha256'])
        self.assertLess(ft['loss_last50'],ct['loss_last50']-.2)
        pred,_=fuse(pools[1][0],pools[1][1],pools[1][2],f)
        self.assertGreater(float(np.mean((pred>=.5)==pools[1][3])),.95)


if __name__=='__main__':unittest.main()
