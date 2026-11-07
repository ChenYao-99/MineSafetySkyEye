# -*- coding: utf-8 -*-
"""Check identity-preserving initialization and source/mine period semantics."""
import ast
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"validation")); sys.path.insert(0,str(ROOT))
import run_native_period_study as study
from run_rgb_joint_study import model_at,CURRENT
from finetune_negatives import norm_patch


class PeriodInformation(unittest.TestCase):
    def test_zero_change_stem_preserves_original_post_function(self):
        previous=study.FUSION; study.FUSION="post_change"
        self.addCleanup(lambda:setattr(study,"FUSION",previous))
        # Four- and eight-channel CUDA stems may select different TF32 kernels.
        # Verify the algebraic initialization in strict FP32 instead of confusing
        # GPU rounding differences with a changed source function.
        old_cudnn_tf32=torch.backends.cudnn.allow_tf32
        old_matmul_tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        self.addCleanup(lambda:setattr(torch.backends.cudnn,"allow_tf32",old_cudnn_tf32))
        self.addCleanup(lambda:setattr(torch.backends.cuda.matmul,"allow_tf32",old_matmul_tf32))
        old=model_at(CURRENT).eval(); new=study.period_model().eval()
        raw=np.load(ROOT/"experiments/native_boundary_20261002/mine_val_x.npy")[:2]
        x=torch.as_tensor(raw,device=study.DEVICE)
        change=torch.full_like(x,123)
        with torch.no_grad():
            expected=old(x); actual=new(torch.cat([x,change],1))
        torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-4)
        self.assertTrue((new.encoder.conv1.weight[:,4:]==0).all())
        del old,new

    def test_real_period_features_use_actual_dates_and_zero_identity_changes(self):
        previous=study.FUSION; study.FUSION="post_change"
        self.addCleanup(lambda:setattr(study,"FUSION",previous))
        features=study.observed_pair("val")
        self.assertEqual(features.shape[1],8)
        self.assertTrue(np.any(features[:,4:]>0))
        post=np.load(ROOT/"experiments/native_boundary_20261002/mine_val_x.npy")
        np.testing.assert_allclose(features[:,:4],post,atol=.001)
        self.assertGreaterEqual(float(features[:,4:].min()),0)
        self.assertLessEqual(float(features[:,4:].max()),255)
        alignment=json.loads((study.CASE/"period_alignment.json").read_text(encoding="utf-8"))
        self.assertTrue(alignment["raw_pair_gate_pass"])

    def test_RGB_proxy_has_no_measured_NIR_or_NIR_change(self):
        x=np.random.default_rng(0).normal(size=(2,8,32,32)).astype(np.float32)
        view=x.copy(); view[:,3]=view[:,2]; view[:,7]=view[:,6]
        np.testing.assert_array_equal(view[:,3],view[:,2])
        np.testing.assert_array_equal(view[:,7],view[:,6])
        # Check the actual native-validation path performs both replacements.
        tree=ast.parse((ROOT/"validation/run_native_period_study.py").read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_validation")
        text=ast.unparse(fn)
        self.assertIn("view[:, 3] = view[:, 2]",text)
        self.assertIn("view[:, 7] = view[:, 6]",text)

    def test_new_test_candidates_still_have_no_final_run(self):
        dataset=ROOT/"datasets/mine_benchmark_20261002"
        self.assertFalse((dataset/"final_evaluation_once.json").exists())
        reg=json.loads((dataset/"mine_case_registry.json").read_text(encoding="utf-8"))
        for eid in ("CL0002","KG0001"):
            self.assertEqual(reg["cases"][eid]["role"],"reserved_final_test")
            self.assertEqual(reg["cases"][eid]["model_predictions_run"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)
