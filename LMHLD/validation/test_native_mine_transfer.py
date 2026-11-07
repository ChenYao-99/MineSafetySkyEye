# -*- coding: utf-8 -*-
"""Check actual mine-level isolation, image identities and conservative label support."""
import json
import unittest
from pathlib import Path
import numpy as np
import cv2
import rasterio
from mine_data_protocol import DATA, assert_development_cases, verify_assets
from prepare_native_mine_folds import OUT, prepare


class NativeTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=prepare()
        cls.plan=json.loads((OUT/'training_protocol.json').read_text(encoding='utf-8'))

    def test_complete_mine_roles_and_fixed_checkpoint(self):
        self.assertEqual(self.plan['checkpoint_rule'],'FIXED FINAL EPOCH15 for every run, no mining/held-mine metric selection')
        self.assertEqual({(r['train_mine'],r['held_development_mine']) for r in self.plan['folds']},
                         {('Amyntaio','LasCruces'),('LasCruces','Amyntaio')})
        self.assertFalse(self.data['held_mine_used_for_checkpoint_selection'])
        for r in self.plan['folds']:
            self.assertNotEqual(r['train_mine'],r['held_development_mine'])
            assert_development_cases([r['train_mine']])
        for eid in ('CL0002','KG0001'):
            with self.assertRaises(ValueError): assert_development_cases([eid])
        self.assertFalse((DATA/'final_evaluation_once.json').exists())
        self.assertTrue(verify_assets())

    def test_las_reference_support_and_optical_unknowns(self):
        folder=DATA/'development/LasCruces'
        arrays=[]
        for name in ('partial_labels.tif','label_confidence.tif','published_accident_reference.tif','post_SCL_10m.tif'):
            with rasterio.open(folder/name) as src: arrays.append(src.read(1))
        y,w,truth,scl=arrays
        self.assertEqual(set(np.unique(y)),{0,1,255})
        self.assertTrue(np.all(w[y==255]==0))
        self.assertTrue(np.all(w[y==1]==np.float32(.6)))
        self.assertTrue(np.all(w[y==0]==np.float32(.25)))
        self.assertFalse(np.any(y[np.isin(scl,[0,1,8,9,10,11])]!=255))
        distance=cv2.distanceTransform((truth>0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        self.assertGreaterEqual(float(distance[y==1].min()),5)
        self.assertFalse(np.any((y==0)&(truth>0)))

    def test_actual_dates_bands_and_common_grid(self):
        folder=DATA/'development/LasCruces'
        images=[]
        for period,date in (('pre','2019-01-07'),('post','2019-01-27')):
            with rasterio.open(folder/f'{period}_BGRN_10m.tif') as src:
                self.assertEqual(src.descriptions,('B02','B03','B04','B08'))
                self.assertEqual(src.dtypes,('uint16',)*4)
                self.assertEqual(src.tags()['acquisition_datetime'][:10],date)
                self.assertEqual(src.res,(10,10))
                images.append((src.crs,src.transform,src.shape))
        self.assertEqual(images[0],images[1])
        audit=json.loads((folder/'period_alignment.json').read_text(encoding='utf-8'))
        self.assertTrue(audit['raw_pair_gate_pass']); self.assertFalse(audit['raw_images_warped'])

    def test_published_reference_limits_and_holdout_ground(self):
        folder=DATA/'development/LasCruces'
        ref=json.loads((folder/'label_reference.json').read_text(encoding='utf-8'))
        audit=json.loads((folder/'source_review/registration_diagnostic.json').read_text(encoding='utf-8'))
        self.assertFalse(ref['field_or_independent_expert_review'])
        self.assertEqual(ref['model_predictions_during_labeling'],0)
        self.assertTrue(audit['registration_gate_pass'])
        self.assertGreaterEqual(audit['heldout_matches'],20)
        self.assertGreaterEqual(audit['heldout_within_3px_fraction'],.8)
        self.assertTrue(audit['not_independent_absolute_geological_boundary_accuracy'])

    def test_training_windows_have_positive_and_negative_support(self):
        for eid in self.data['case_ids']:
            case=self.data['cases'][eid]
            x=np.load(OUT/f'{eid}_x.npy'); y=np.load(OUT/f'{eid}_y.npy'); w=np.load(OUT/f'{eid}_weight.npy')
            self.assertEqual(x.shape,(len(case['train_windows']),4,32,32))
            self.assertGreater(case['positive_windows'],0); self.assertGreater(case['negative_windows'],0)
            self.assertFalse(np.any(w[y==255]!=0))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(case['overlapping_windows_are_not_independent_cases'])

    def test_raw_native_inference_matches_training_preprocessing(self):
        # Actual fixed weights and uint16-origin data: catch silently feeding raw DN
        # to a model trained on normalized windows even when output shapes are valid.
        from evaluate_native_mine_transfer import native_probabilities
        from finetune_negatives import norm_patch
        from run_rgb_joint_study import CURRENT
        from detector import LandslideDetector
        with rasterio.open(DATA/'development/LasCruces/post_BGRN_10m.tif') as src:
            raw=src.read(window=rasterio.windows.Window(64,64,32,32)).astype(np.float32)
        detector=LandslideDetector(str(CURRENT),patch_size=32,threshold=.5)
        for proxy in (False,True):
            channels=raw.copy()
            if proxy: channels[3]=channels[2]
            expected=detector.predict_batch(norm_patch(channels[None]))[0]
            actual=native_probabilities(detector,raw.transpose(1,2,0),copy_red=proxy)
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)


if __name__=='__main__': unittest.main()
